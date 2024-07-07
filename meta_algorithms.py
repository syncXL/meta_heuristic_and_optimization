import pandas as pd
from variables import *
import os,time
class Alg_General():
    def __init__(self,params,fit_function,goal,n_trials,stop_criteria,verify_params=False,verify_fn = None):
        self.goal = goal
        self.verify_parameter = verify_params
        self.verify_fn = verify_fn
        self.stop_criteria = stop_criteria
        self.params = params
        self.n_trials = n_trials
        self.current_trial_id = 0
        self.run_id = 0
        self.fit_function = fit_function
        self.best_params = [{param : 0 for param in list(self.params.keys())}
            for _ in range(n_trials)]
        self.max_params = np.array([val.max_val for val in list(self.params.values())])
        self.min_params = np.array([val.min_val for val in list(self.params.values())])
        self.history = [{param : [] for param in list(self.params.keys())}
            for _ in range(n_trials)]
        self.history = [ {"run":[]}|hist|{"score":[]} for hist in self.history ]
        self.best_params = [ {"run":0}|best|{"score":0} for best in self.best_params ]

    def evaluate(self,params,show_summary=True):
        current_params_val = {key : value for key,value in zip(self.params.keys(),params)}
        self.run_id +=1
        current_params_val["score"] = self.fit_function(**current_params_val)
        if show_summary:
            self.runtime_summary(params)
            print(f"Score: {current_params_val['score']}")
            print("\n\n")
        current_params_val["run"] = self.run_id
        self.record(**current_params_val)
        absolutely_unused_variable = os.system('cls')
        return current_params_val["score"],current_params_val["run"]

    def record(self,**inst):
        for key,value in inst.items():
            self.history[self.current_trial_id][key].append(value)

    def visualize(self):
        pass

    def is_better(self,old_val,new_val):
        if self.goal == "max":
            return old_val > new_val
        return old_val < new_val

    def terminate(self):
        if self.stop_criteria[0] == 'early':
            if (self.run_id - self.best_params[self.current_trial_id]["run"]) - 1 > self.stop_criteria[1]:
                return True
            return False
        elif self.stop_criteria[0] == "no_iterations":
            if self.run_id> self.stop_criteria[1]:
                return True
            return False

    def formatted(self,col_val,max_len):
        col_len = len(col_val)
        if col_len > max_len:
            return str(col_val)[:3] + "..."
        return str(col_val) + " ".join(["" for _ in range(max_len-col_len+1)])

    def runtime_summary(self,params):
        current_params_val = {key : params[ind] for ind,key in enumerate(list(self.params.keys()))}
        first = len(self.history[self.current_trial_id]['score']) >= 1
        print(f"Trial ID: {self.current_trial_id}")
        print(f"Best Score so far: {self.best_params[self.current_trial_id]["score"]}")
        print(f"Previous Run Score: {self.history[self.current_trial_id]["score"][-1] if first else "--" }")
        header = ["Hyperparameter",f"Best value ({self.best_params[self.current_trial_id]["run"]})",f"Previous value ({self.run_id-1 if first else "--"})",f"Current Value ({self.run_id})"]
        header_len = [len(head) for head in header]
        print(f"{" | ".join(header)}")
        for hp in list(self.params.keys()):
            prev = self.history[self.current_trial_id][hp][-1] if first else "---"
            x = [self.params[hp].name,self.best_params[self.current_trial_id][hp],prev,current_params_val[hp]]
            row = [self.formatted(str(fmt_str),max_l) for fmt_str,max_l in zip(x,header_len)]
            print(f"{" | ".join(row)}")

    def final_summary(self):
        print("Best value for each Hyperparameter")
        columns = ["Trial No"]
        columns.extend([x.name for x in list(self.params.values())])
        columns.append("Score")
        print(f"{" | ".join(columns)}")
        c_lens = [len(col) for col in columns]
        for trial in range(self.n_trials):
            x = [trial]
            x.extend([self.best_params[trial][key] for key in (list(self.params.keys()))])
            # print(self.best_params[trial]["score"])
            # time.sleep(100)
            try:
                x.append(self.best_params[trial]["score"][0])
            except:
                x.append(self.best_params[trial]["score"])
            row =[self.formatted(str(col),c_l) for col,c_l in zip(x,c_lens)]
            print(f"{" | ".join(row)}")
    
    def best_parameter(self,n_trial=0):
        return {self.best_params[n_trial][key] for key in (list(self.params.keys()))}

    def params_to_numpy(self):
        return np.array([val.val for val in list(self.params.values())])

    def best_params_to_numpy(self):
        return np.array([self.best_params[self.current_trial_id][key] for key in list(self.params.keys())])

    def verify_params(self,params,reshape=True):
        if (self.verify_parameter == None) | (self.verify_parameter == "Both"):
            query = (params >= self.min_params) & (params <= self.max_params)
            if reshape:
                sh = params.shape[1]
            else:
                    x = np.full((1,query.shape[0]),1)
                    for col in range(query.shape[1]):
                        x *= query[:,col].astype(np.int32)
                    x = x.astype(bool)
                    query = x.ravel()
            if reshape:
                return params[query].reshape(-1,sh)
        if self.verify_parameter:
            return self.verify_fn(params[query])
        else:
            return params[query]

class PatternSearch(Alg_General):
    def __init__(self,
        params : dict,
        fit_function,
        goal : str,
        n_trials : int,
        stop_criteria : tuple,
        alpha :float,
        int_step : int = 2,
        float_step : float = 0.1,
        int_step_lim : int = 1,
        float_step_lim : float = 0.1,
        decrease_rate : float = 0.2,
        move_type : str = 'MADS'):

        self.alpha = alpha
        self.decrease_rate = decrease_rate
        self.def_mu = []
        self.mu_lim = []
        for val in list(params.values()):
            if val.return_type() == "int":
                self.def_mu.append(int_step)
                self.mu_lim.append(int_step_lim)
            elif val.return_type() == "float":
                self.def_mu.append(float_step)
                self.mu_lim.append(float_step_lim)
        self.def_mu = np.array(self.def_mu)
        self.mu_lim = np.array(self.mu_lim)
        self.mu = self.def_mu.copy()

        self.move_type = move_type
        super().__init__(params, fit_function, goal, n_trials, stop_criteria)

    def update_best(self,params,score,runid):
        # time.sleep(100)
        self.best_params[self.current_trial_id]["run"] = runid
        self.best_params[self.current_trial_id]["score"] = score
        for key,val in zip(list(self.params.keys()),params):
            self.best_params[self.current_trial_id][key] = val
    
    def explore(self,base):
        cur_comb = base.copy()
        princ_axes = self.mu * np.identity(self.mu.shape[0])
        potent_best = princ_axes + cur_comb
        if self.move_type == "GPS":
            princ_axes = -1 * princ_axes
            potent_best = np.hstack((potent_best,princ_axes + cur_comb)).transpose()
        elif self.move_type == "MADS":
            princ_axes = np.full((self.mu.shape[0],1),-1) * self.mu
            # print(potent_best.shape)
            potent_best = np.hstack((potent_best,(princ_axes + cur_comb))).transpose()
        # print(potent_best)
        if potent_best.shape[1] > 1:
            potent_best = self.verify_params(potent_best,reshape=False)
        else:
            potent_best = self.verify_params(potent_best,reshape=False)
        scores,ids = [],[]
        for params in potent_best:

            score,id = self.evaluate(params)
            scores.append(score)
            ids.append(id)
        scores = np.array(scores,np.float32).reshape(-1,1)
        ids = np.array(ids,np.int32).reshape(-1,1)
        return potent_best,scores,ids

    def reset_mesh(self):
        self.mu = self.def_mu.copy()
    
    def verify_step(self,step):
        for ind,val in enumerate(list(self.params.values())):
            if isinstance(val.val,np.int32):
                step[ind] = np.floor(step[ind])
        return step

    def get_pattern_point(self,prev_base):
        base_p = self.best_params_to_numpy()
        pattern = self.alpha * (base_p - prev_base)
        patt_point = prev_base + pattern
        patt_point = patt_point.T
        patt_point = patt_point.reshape(1,-1)
        patt_point_verified = self.verify_params(patt_point,False)
        if patt_point_verified.shape[0] == 0:
            return patt_point
        return patt_point_verified

    def main(self,start_point,base_type):
        absolutely_unused_variable = os.system('cls')
        if base_type == "explore":
            print("Exploratory Move")
        else:
            print("Pattern Move")
        combinations,scores,comb_id =self.explore(start_point)
        better_scores = scores > self.best_params[self.current_trial_id]["score"]
        better_scores = better_scores.ravel()
        if scores[better_scores].shape[0] > 0:
            if base_type == "explore":
                self.reset_mesh()
            prev_base = self.best_params_to_numpy()
            trial_comb,trial_score,trial_id = combinations[better_scores][0],scores[better_scores][0],comb_id[0]
            trial_comb = trial_comb.tolist()
            try:
                len(trial_comb)
            except TypeError:
                trial_comb = [trial_comb]
            self.update_best(trial_comb,trial_score,trial_id)
            if not self.terminate():

                new_trial_point = self.get_pattern_point(prev_base)
                self.main(new_trial_point,"pattern")
        else:
            self.decrease_mesh()
            if not self.terminate():
                self.main(self.best_params_to_numpy(),"explore")
    
    def terminate(self):
        if self.stop_criteria[0] == "mesh":
            return all((self.mu - self.mu_lim) < 0)
        return super().terminate()
    
    def decrease_mesh(self):
        self.mu = self.mu - (self.decrease_rate * self.mu)
        self.mu = self.verify_step(self.mu)
    
    def run(self):
        for trial in range(self.n_trials):
            print("Exploration Move")
            score,id = self.evaluate(self.params_to_numpy())
            self.update_best(self.params_to_numpy(),score,id)
            self.main(self.best_params_to_numpy(),"explore")
            self.current_trial_id +=1
            self.run_id = 0
            self.mu = self.def_mu
            for val in self.params.values():
                val.set_random()
        absolutely_unused_variable = os.system('cls')
        self.final_summary()

class GeneticAlgorithm(Alg_General):
    def __init__(self,
                population : int,
                n_parents : int,
                n_generation : int,
                params,
                fit_function,
                goal,
                n_trials,
                stop_criteria,
                c_method : str,
                m_method : str,
                n_best : int = 10,
                p_cross : float = 0.6,
                p_mutate : float = 0.3,
                seed :int = 42,
                parent_method : str = 'proportion',
                z : int = 1,
                t_sel : int = 3,
                verify_params=False,
                verify_fn= None):
        self.seed = seed
        self.population = population
        self.n_children = self.population - n_parents
        self.p_cross = p_cross
        self.p_mutate = p_mutate
        self.n_generation = n_generation
        self.t_sel = t_sel
        self.z = z
        self.n_parents = n_parents
        self.p_method = [self.proportion,self.rank,self.tournament][["proportion","rank","tournament"].index(parent_method)]
        self.cross = [self.one_cross,self.two_cross,self.uniform_cross][["one","two","uniform"].index(c_method)]
        self.mutate = [self.mutate_uniform,self.mutate_n_uniform][["uniform","non_uniform"].index(m_method)]
        super().__init__(params, fit_function, goal, n_trials, stop_criteria, verify_params, verify_fn)
        self.history = [ hist|{"Generation":[]} for hist in self.history]
        self.best_params = [ best|{"Generation":[]} for best in self.best_params]

    def generate_chromosomes(self):
        z = np.array([[param.randomize() for param in self.params.values()] for _ in range(self.population)])
        if self.verify_parameter:
            z = self.verify_fn(z)
        return z

    def evaluate(self, params, show_summary=True):
        print(f"Generation : {self.generation}")
        sc,run_id =  super().evaluate(params, show_summary)
        self.history[self.current_trial_id]['Generation'].append(self.generation)
        return sc,run_id
    
    def main(self,chromosome,gen = 1):
        self.generation = gen
        self.chromosomes = chromosome
        fit_val = np.array([self.evaluate(chromosome) for chromosome in self.chromosomes])
        self.parents = self.chromosomes[self.p_method(fit_val)]
        self.children = self.crossover()
        self.mutation()
        self.chromosomes = np.vstack([self.parents,self.children])

        if self.generation - self.n_generation != 0:
            self.generation += 1
            self.main(self.chromosomes,self.generation)

    def proportion(self,fit_val):
        pr_fit = fit_val[:,:1] / fit_val[:,:1].sum()
        pr_ind = np.arange(len(pr_fit)).reshape(-1,1)
        selected_list = []
        if self.goal == 'min':
            pr_fit = (1 - pr_fit)
            pr_fit = pr_fit / pr_fit.sum()
        np.cumsum(pr_fit,out=pr_fit)
        while len(selected_list) != self.n_parents:
            rnd = np.random.random()
            selected = pr_ind[pr_fit >= rnd][0]
            if selected not in selected_list:
                selected_list.append(selected)
        return np.array(selected_list)

    def rank(self,fit_val):
        rank_fit = np.sort(fit_val,axis=0)
        rank_ind = np.arange(start=1,stop=rank_fit.shape[0] +1)
        if self.goal == 'min':
            rank_fit[:,0] = rank_fit[::-1,0]
        pr_rank = ((2 - self.z) / rank_fit.shape[0]) + (((2 * rank_ind) * (self.z -1)) / (rank_fit.shape[0] * (rank_fit.shape[0] -1)))
        pr_rank = pr_rank / pr_rank.sum()
        np.cumsum(pr_rank,out=pr_rank)
        ind = np.full(shape=pr_rank.shape[0],fill_value=False)
        while ind.sum() != self.n_parents:
            rnd = np.random.random()
            selected = rank_ind[pr_rank >= rnd][0]
            ind[selected-1] = True
        return ind

    def tournament(self,fit_val):
        fit_val = fit_val[:,:1]
        h = fit_val.shape[0] -1
        ind = np.full(shape=h+1,fill_value=False)
        while ind.sum() != self.n_parents:
            inds = np.array(self.pair(maxind=h,n_item=self.t_sel))
            chosen = fit_val[:,:1][inds]
            winner = chosen.max()
            if self.goal == 'min':
                winner = chosen.min()
            winner_ind = inds[(chosen == winner).ravel()][0]
            ind[winner_ind] = True
        return ind

    def pair(self,maxind,n_item = 2):
        inds = []
        while len(inds) != n_item:
            rnd_num = np.random.randint(low=0,high=maxind)
            if rnd_num not in inds:
                inds.append(rnd_num)
        return inds

    def crossover(self):
        rnd_prs = np.random.random(size=self.parents.shape[0])
        n_pairs = int(self.n_children)
        cross_over_parents = self.parents[rnd_prs < self.p_cross]
        if cross_over_parents.shape[0] < n_pairs+1:
            cross_over_parents = self.parents.copy()
        couples = np.array([self.pair(cross_over_parents.shape[0] - 1) for _ in range(n_pairs) ])
        children = []
        for couple in couples:
            child_1,child_2 = self.cross(cross_over_parents[couple[0],:],cross_over_parents[couple[1],:])
            children.append(child_1)
            children.append(child_2)
        return np.array(children)[:self.n_children]

    def one_cross(self,*parent):
        rnd_pt = np.random.randint(low=1,high=parent[0].shape[0]-1)
        return np.hstack([parent[0][:rnd_pt],parent[1][rnd_pt:]]),np.hstack([parent[1][:rnd_pt],parent[0][rnd_pt:]])
    
    def two_cross(self,*parent):
        rnd_pts = self.pair(parent[0].shape[0]-1)
        lb = rnd_pts[0] if rnd_pts[0] < rnd_pts[1] else rnd_pts[1]
        ub = rnd_pts[1] if rnd_pts[1] > rnd_pts[0] else rnd_pts[0]
        child_1 = np.hstack([parent[1][:lb],parent[0][lb:ub]])
        child_1 = np.hstack([child_1,parent[1][ub:]])
        child_2 = np.hstack([parent[0][:lb],parent[1][lb:ub]])
        child_2 = np.hstack([child_2,parent[0][ub:]])
        return child_1,child_2
    
    def uniform_cross(self,*parent):
        rnd_pts = self.pair(parent[0].shape[0])
        children = (parent[0].copy(),parent[1].copy())
        for ind,child in enumerate(children):
            child[rnd_pts[0]] = parent[ind-1][rnd_pts[0]]
            child[rnd_pts[1]] = parent[ind-1][rnd_pts[1]]
        return children

    def mutation(self):
        rnd_prs = np.random.random(size=self.parents.shape[1])
        var_mutate = np.arange(self.parents.shape[1])[rnd_prs < self.p_mutate]
        if var_mutate.shape[0] == 0:
            return self.children
        else:
            self.mutate(var_mutate)

    def mutate_uniform(self,var_mutate):
        the_variables = [list(self.params.keys())[x] for x in var_mutate]
        for ind,var in zip(var_mutate,the_variables):
            self.children[:,ind] = np.array([[self.params[var].randomize()] for _ in range(self.children.shape[0])]).reshape(-1)
        if self.verify_parameter:
            self.verify_fn(self.children)

    def mutate_n_uniform(self,var_mutate):
        the_variables = [list(self.params.keys())[x] for x in var_mutate]
        for ind,var in zip(var_mutate,the_variables):
            z = ((self.n_generation - self.generation)/ self.n_generation)
            d = self.children[:,ind] * z
            d = d.astype(np.int32) if self.params[var].dtype != 'float' else d
            var_rnd = np.random.random(self.children[:,ind].shape)
            change = ((2 * d) * var_rnd) - d
            change = change.astype(np.int32) if self.params[var].dtype != 'float' else change
            mutated_var = self.children[:,ind] + change
            invalids = (mutated_var < self.params[var].min_val) | (mutated_var > self.params[var].max_val)
            mutated_var[invalids] = self.children[:,ind][invalids]
            if self.verify_parameter:
                mutated_var[invalids] = self.verify_fn(self.children[:,ind][invalids])
            self.children[:,ind] = mutated_var

    def store_best(self):
        best_id = np.argmax(self.history[self.current_trial_id]["score"])
        if self.goal == 'min':
            best_id = np.argmin(self.history[self.current_trial_id]["score"])
        
        for key in list(self.best_params[self.current_trial_id].keys()):
            self.best_params[self.current_trial_id][key] = self.history[self.current_trial_id][key][best_id]

    def run(self):
        for _ in range(self.n_trials):
            self.main(self.generate_chromosomes())
            self.store_best()
            self.current_trial_id += 1
            self.run_id = 0
        self.final_summary()

    def return_history(self,trial_no):
        return pd.DataFrame(self.history[trial_no - 1])


    


    # def run(self):
    #     for 

# def area(**params):
#     return (480 * params['l']) - (92 * params['l']**2) + (4 * params['l']** 3)

# params = {
#     'l' : Var_float(name='Length',max_val = 8,min_val=0),
#     'w' : Var_int(name='Width',max_val = 40,min_val=0),
#     'h' : Var_int(name='Height',max_val = 3,min_val=0),
#     # 'features' : Var_Choice(name='features',choices=[1,2,3,4,5,6,7])
# }
# galg = GeneticAlgorithm(50,25,10,params,area,'min',1,("early",1),parent_method='proportion',z=1.5,c_method='uniform',m_method='non_uniform')
# galg.run()

# patten_alg = PatternSearch(params,area,'max',2,("early",1),alpha=2,int_step=2,int_step_lim=1,decrease_rate=0.3,move_type="MADS")

# for p,k in params.items():
#     print(f"{p} : {k.randomize()}")
# patten_alg.run()