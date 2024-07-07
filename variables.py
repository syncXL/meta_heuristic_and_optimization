import numpy as np
import random
from numpy.random.mtrand import randint as randint

class Var_General():
    def __init__(self,name,dt,**others):
        self.dtype = dt
        self.name = name
        self.random_state = np.random.randint(1,10e4)
        np.random.seed(self.random_state)
        if dt=='float':
            # self.coeff = others["coeff"]
            self.max_val = np.array(others["max_val"],dtype=np.float32)
            self.min_val = np.array(others["min_val"],dtype=np.float32)
        elif dt == 'int':
            # self.coeff = others["coeff"]
            self.max_val = np.array(others["max_val"],dtype=np.int32)
            self.min_val = np.array(others["min_val"],dtype=np.int32)
        elif dt == 'choice':
            self.max_val = len(others["choices"])
            self.min_val = 0
            self.choices = others["choices"]
            self.choices_int = np.arange(len(others["choices"]))

    def summary(self,val=None,step=None,mov=None):
        if self.dtype != 'choice':
            print(f"Name : {self.name},\nUpper Bound : {self.max_val},\nLower Bound : {self.min_val},\nCurrent Value : {val},\nData Type : {self.dtype}\nRandom_State : {self.random_state},\nMove Type: {mov},\nStep : {step}")
        else:
            print(f"Name : {self.name},\nData Type : {self.dtype},\nOptions : {self.choices},\nChosen Option : {val},\nRandom_State : {self.random_state}")
    
    def return_type(self):
        return self.dtype
class Var_int(Var_General):
    def __init__(self,def_val="random",mov='half',**attr):
        super().__init__(dt='int',**attr)
        if def_val == "random":
            self.set_random()
        else:
            self.val = np.array(def_val,dtype=np.float32)
        self.mov = mov
        self.step = self.set_step()
    def set_step(self):
        return np.array(1,dtype=np.float32)
    def randomize(self):
        return np.random.randint(self.min_val,self.max_val)
    def set_random(self):
        self.val = self.randomize()
    def summary(self):
        return super().summary(self.val, self.step, self.mov)

class Var_float(Var_General):
    def __init__(self, def_val="random",mov="half",**attr):
        super().__init__(dt='float',**attr)
        if def_val == "random":
            self.set_random()
        else:
            self.val = np.array(def_val,dtype=np.float32)
        self.mov = mov
        self.step = self.set_step()
    def set_step(self):
        return np.array(1,dtype=np.float32)
    def randomize(self):
        return ((self.max_val - self.min_val) * np.random.random_sample(1) + self.min_val)[0]
    def set_random(self):
        self.val = self.randomize()
    def summary(self):
        return super().summary(self.val, self.step, self.mov)

class Var_Choice(Var_General):
    def __init__(self,def_val="random",**attr):
        super().__init__(dt='choice',**attr)
        if def_val == "random":
            self.set_random()
        else:
            self.def_val_str = def_val
            self.def_val_int = self.choices.index(self.def_val_str)
    def randomize(self):
        return np.random.choice(self.choices_int,size=1)[0]
    def set_random(self):
        self.def_val_int = self.randomize()
        self.def_val_str = self.choices[self.def_val_int]
    def summary(self):
        return super().summary(self.def_val_str)
