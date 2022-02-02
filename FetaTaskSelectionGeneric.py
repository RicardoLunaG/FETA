import numpy as np
import tensorflow as tf
from garage import wrap_experiment
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.envs import normalize


#import tensorflow_probability as tfp
#tf.compat.v1.disable_eager_execution()

selected_tasks = []
tasks_source = {} #Parameters of the available training tasks

def evaluate(source_task, EPOCS=5,NUMBER_EPISODES = 10):
        
    evaluating_results = []
    number_of_task_episodes = 200
    print(source_task)
    @wrap_experiment(log_dir="./Saved_Models/HalfCheetahRGA/Test"+str(source_task),snapshot_mode="none",snapshot_gap= 0,use_existing_dir=False,name="Task"+str(source_task))
    def evaluate_target_tasks(ctxt=None,targetTask=0):
        last_rewards_total = []
        for epoc in range(EPOCS):
            
            tf.compat.v1.reset_default_graph()
            
            with tf.compat.v1.Session() as sess:
                
                with LocalTFRunner(snapshot_config=ctxt,sess=sess) as runner:
                    
                    saved_dir = "./Saved_Models/HalfCheetahRGA/Test"+str(targetTask)

                    env = TfEnv(normalize(DesiredGymEnv(task=tasks_source[str(source_task)]))) #Should work with any domain defined following the standard Open AI Gym template
                    

                    runner.restore(from_dir=saved_dir,env = env)

                                       
                    runner.resume(n_epochs=runner._stats.total_epoch+(NUMBER_EPISODES),batch_size=10)
                    reward_total = 0 
                    state = env.reset()                    
                    for _ in range(number_of_task_episodes): #Policy Evaluation, evaluating on return. Can be change for goal base domains.
                        
                        action,_ = runner._policy.get_action(state)
                                                
                        state, reward, _, _ = env.step(action)
                        
                        reward_total += reward
                    last_rewards_total.append(reward_total) 
            sess.close()
        return last_rewards_total
    for selected in selected_tasks:
        results = evaluate_target_tasks(targetTask=str(selected))
        evaluating_results.append(np.average(results))
    
    return succesful_transfer(evaluating_results)

def succesful_transfer():
    "Define Succesful Transfer Function, Domain Dependent"
    pass
def main():
    
    global selected_tasks
 

    valid_tasks = np.arange(1,41)

    total_selected_tasks = []
    print(valid_tasks)
    num_runs = 100 #Number of runs in which Feta is run, each time with a random order of tasks.
    
    for i in range(num_runs):
        np.random.shuffle(valid_tasks)
        selected_tasks = []
        for seed_task in valid_tasks:
            
            if len(selected_tasks) == 0:
                selected_tasks.append(seed_task)
                
            elif evaluate(seed_task):
                selected_tasks.append(seed_task)
            else:
                continue
            

        
        selected_tasks.sort()
        if selected_tasks not in total_selected_tasks:
            total_selected_tasks.append(selected_tasks)

        
        print("ACEPTED TASKS")
        print(total_selected_tasks)

        with open("selected_tasks_hc2.txt", "w") as f:
            for s in total_selected_tasks:
                f.write(str(s) +"\n")
            


if __name__ == '__main__':

    main()