import os, time, shutil, yaml
from collections import defaultdict

class Counter:
    def __init__(self, *initial_fields):
        self._start_time = time.time()
        self._internal = {}
        for name in initial_fields:
            self._internal[name] = 0
        self._internal["time"] = 0.
    def add_fields(self, *fields):
        time = self._internal["time"]
        del self._internal["time"]
        for name in fields:
            self._internal[name] = 0
        self._internal["time"] = time
    def increment(self, name, amount=1):
        self._internal[name] += amount
    def loop(self, name, total_or_generator=None, count_on_break=True):
        self.increment(name, 0)
        if total_or_generator is None:
            yield from self._infinite_loop(name, count_on_break)
        elif isinstance(total_or_generator, int):
            yield from self._int_loop(name, total_or_generator, count_on_break)
        else:
            yield from self._generator_loop(name, total_or_generator, count_on_break)
    def _infinite_loop(self, name, count_on_break=True):
        try:
            i = 0
            while True:
                yield i
                i += 1
                self.increment(name)
        except GeneratorExit:
            if count_on_break: self.increment(name)
    def _int_loop(self, name, total, count_on_break=True):
        try:
            for i in range(total):
                yield i
                self.increment(name)
        except GeneratorExit:
            if count_on_break: self.increment(name)
    def _generator_loop(self, name, generator, count_on_break=True):
        try:
            for item in generator:
                yield item
                self.increment(name)
        except GeneratorExit:
            if count_on_break: self.increment(name)
    def __getattr__(self, item):
        if item == "_internal": raise AttributeError("why does _internal not exist?")
        if item == "time": self.update_time()
        if item in self._internal: return self._internal[item]
        return super().__getattr__(item)
    def update_time(self):
        self._internal["time"] = time.time() - self._start_time
    def keys(self):
        return self._internal.keys()
    def view(self, ignore_zero=False):
        self.update_time()
        return ', '.join([f"{key}={round(getattr(self,key))}" for key in self.keys() if (not ignore_zero or getattr(self, key) > 0)])
    def dict(self):
        self.update_time()
        return self._internal


def to_string_for_csv(x):
    if isinstance(x, list):
        return ';'.join([str(item) for item in x])
    else:
        return str(x)

class Logger:
    def __init__(self, loc, write_rate=30):
        if loc is not None:
            self.loc = loc
            if os.path.exists(self.loc):
                shutil.rmtree(self.loc)
            os.makedirs(self.loc, exist_ok=True)
            self.fake = False
        else:
            self.fake = True
        self.write_rate = write_rate
        self.components = {}
        self.cache = {}
        self.write_count = defaultdict(int)
        self.last_write_time = time.time()

    def write(self, name, _flush=False, _subsample_rate=1, **contents):
        if self.fake: return
        if name not in self.components:
            self.components[name] = list(contents.keys())
            self.cache[name] = ""
            with open(os.path.join(self.loc, f"{name}.csv"), "a") as f: f.write(",".join(str(x) for x in self.components[name]))
        if self.write_count[name] % _subsample_rate == 0:
            self.cache[name] += "\n" + ",".join(to_string_for_csv(contents[key]) for key in self.components[name])
        self.write_count[name] += 1
        if _flush or time.time() - self.last_write_time > self.write_rate: self._write_all()

    def _write_all(self):
        for name, cached_str in self.cache.items():
            with open(os.path.join(self.loc, f"{name}.csv"), "a") as f: f.write(cached_str)
            self.cache[name] = ""
        self.last_write_time = time.time()

    def write_hps(self, hps):
        if self.fake: return
        with open(os.path.join(self.loc, f"hps.yaml"), "w") as f:
            f.write(yaml.dump(hps, allow_unicode=True))
