import torch

class Data:
    def __init__(self, **kwargs):
        for k,v in kwargs.items(): setattr(self, k, v)

class ShortTermMemory:
    def __init__(self, history_n, spaces, memory_batch=None, device="cpu"):
        self.history_n = history_n
        self.spaces = spaces
        self.memory_batch = memory_batch
        self.device = device
        if self.memory_batch is not None: self.initialize()

    def initialize(self):
        for key, (shape, dtype) in self.spaces.items():
            if len(shape) > 0: t = torch.zeros((self.memory_batch, self.history_n*shape[0],) + shape[1:], dtype=dtype, device=self.device)
            else:              t = torch.zeros((self.memory_batch, self.history_n,), dtype=dtype, device=self.device)
            setattr(self, key, t)

    def add_data(self, data):
        if self.memory_batch is None:
            for x in data:
                if x is not None:
                    self.memory_batch = x.shape[0]
                    break
            self.initialize()
        for key in self.spaces.keys():
            # torch.roll(getattr(self, key), -1, 1)
            if len(self.spaces[key][0]) > 0:
                roll_amount = self.spaces[key][0][0]
                getattr(self, key)[:,:-roll_amount] = getattr(self, key)[:,roll_amount:]
                getattr(self, key)[:,-roll_amount:] = getattr(data, key)
            else:
                getattr(self, key)[:,:-1] = getattr(self, key)[:,1:]
                getattr(self, key)[:,-1] = getattr(data, key)

    def __call__(self, data):
        self.add_data(data)
        return self

class LongTermMemory:
    def __init__(self, spaces, memory_batch=1, memory_size=1000000, precontext_rows=0, buffer_rows=None, device="cpu"):
        self.spaces = spaces
        self.device = device
        self.memory_batch = memory_batch
        self.memory_size = memory_size
        self.buffer_rows = buffer_rows
        self.precontext_rows = precontext_rows
        self.buffer_rows = buffer_rows if buffer_rows is not None else self.memory_size // 10
        for key, (shape, dtype) in self.spaces.items():
            t = torch.zeros((self.memory_size + self.buffer_rows, self.memory_batch,) + shape, dtype=dtype, device=self.device)
            setattr(self, key, t)

        self.start_row = precontext_rows
        self.rows_filled = 0

    @property
    def count(self): return self.rows_filled * self.memory_batch

    def send_to(self, device):
        self.device = device
        for key in self.spaces:
            setattr(self, key, getattr(self, key).to(self.device))

    @staticmethod
    def basic_fetcher(key, offset=0):
        def fetch(data, idx):
            return getattr(data, key)[idx[:,0] + offset, idx[:,1]]
        return fetch

    def add_data(self, **data):
        if self.start_row + self.rows_filled == self.memory_size + self.buffer_rows:
            for key in self.spaces.keys():
                getattr(self, key)[:self.precontext_rows+self.rows_filled] = getattr(self, key)[self.start_row-self.precontext_rows:self.start_row+self.rows_filled]
            self.start_row = self.precontext_rows
        self.set(self.start_row + self.rows_filled, **data, _creating_new_rows=True)
        if self.rows_filled <= self.memory_size:
            self.rows_filled += 1
        else:
            self.start_row += 1

    def add_fields(self, spaces):
        self.spaces.update(spaces)
        for key, (shape, dtype) in spaces.items():
            t = torch.zeros((self.memory_size + self.buffer_rows, self.memory_batch,) + shape, dtype=dtype, device=self.device)
            setattr(self, key, t)

    def get(self, idx, **fns):
        ## fns should be of the form: {key: (lambda data, idx: tensor)}
        if len(fns) == 0: fns = {key: self.basic_fetcher(key) for key in self.spaces.keys()}
        for key, fn in fns.items():
            if isinstance(fn, str): fns[key] = self.basic_fetcher(fn)
        assert torch.all(idx[:,0] >= self.start_row), f"idx less than start row, {self.start_row}"
        assert torch.all(idx[:,0] < self.start_row + self.rows_filled), f"idx greater than filled rows, {self.start_row + self.rows_filled}"
        assert torch.all(idx[:,1] < self.memory_batch), f"idx greater than parallel rollout number, {self.memory_batch}"
        idx[:,0] = torch.remainder(idx[:,0], self.start_row + self.rows_filled)
        idx[:,1] = torch.remainder(idx[:,1], self.memory_batch)
        return Data(idx=idx, **{key: fn(self, idx) for key, fn in fns.items()})

    def set(self, idx, _creating_new_rows=False, **data):
        if isinstance(idx, int): idx = torch.tensor([[idx, i] for i in range(self.memory_batch)], device=self.device)
        assert torch.all(idx[:,0] < self.memory_size + self.buffer_rows), f"idx greater than allocated rows, {self.allocated_rows}"
        assert torch.all(idx[:,1] < self.memory_batch), f"idx greater than parallel rollout number, {self.memory_batch}"
        if not _creating_new_rows:
            assert torch.all(idx[:,0] < self.start_row + self.rows_filled), f"idx greater than current maximum row, {self.start_row + self.rows_filled}. if this was intentional, specify creating_new_rows=True."
            idx[:,0] = torch.remainder(idx[:,0], self.start_row + self.rows_filled)
        idx[:,1] = torch.remainder(idx[:,1], self.memory_batch)
        for k, v in data.items(): getattr(self, k)[idx[:,0],idx[:,1]] = v

    def idx_iterator(self, batch_size, infinite=False, discard_partial=False, required_postcontext=1):
        batch_size = min(batch_size, self.count - required_postcontext*self.memory_batch)
        epoch_idx = torch.tensor([[idx, rollout_i] for rollout_i in range(self.memory_batch) for idx in range(self.start_row, self.start_row + self.rows_filled - required_postcontext)], device=self.device)
        epoch_idx = epoch_idx[torch.randperm(epoch_idx.shape[0])]
        loc = 0
        new_batch_size = None
        while loc < len(epoch_idx):
            batch_idxs = epoch_idx[loc:loc + batch_size]
            if not (discard_partial and len(batch_idxs) < batch_size and batch_size < len(epoch_idx)):
                new_batch_size = yield batch_idxs
            loc += batch_size
            if new_batch_size != None: batch_size = new_batch_size
            if infinite and loc >= len(epoch_idx):
                if batch_size < self.count - required_postcontext*self.memory_batch:
                    epoch_idx = epoch_idx[torch.randperm(epoch_idx.shape[0])]
                loc = 0
