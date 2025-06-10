from typing import Any
import torch
import os

class Checkpointer:
    def __init__(self, ckpt_path = None):
        self.ckpt_path = ckpt_path
        self.models = {}
        self.ckpt = {}

    def save(self, path_override = None):
        path = path_override or self.ckpt_path
        try:
            os.makedirs(os.path.dirname(path))
        except FileExistsError:
            pass
        models = {}
        for k, v in self.models.items():
            try:
                models[k] = v.state_dict()
            except:
                try:
                    models[k] = v.save()
                except:
                    models[k] = v
        torch.save(models, path)
    
    def register(self, models):
        self.models.update(models)

    def load(self, path_override = None, filter_keys=None):
        path = path_override or self.ckpt_path
        if not os.path.exists(path):
            print(f"no checkpoint at {path} found")
            return False
        print(f"loading checkpoint from {path}")
        self.ckpt = torch.load(path)
        for k, v in self.ckpt.items():
            if (filter_keys is not None) and (k not in filter_keys):
                continue
            try:
                self.models[k].load_state_dict(v)
                print(f"successfully loaded state dict for {k}")
            except:
                try:
                    self.models[k].load(v)
                    print(f"successfully loaded {k}")
                except:
                    self.models[k] = v
                    print(f"successfully loaded {k}")
        return True