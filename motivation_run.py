import os
from multiprocessing import Pool, current_process, Queue
from utils import OPT_PATH
import glob


baseline_DTMI_cmd = 'python main.py --gpu {gpu} --white_box {model} --loss_fn {loss_fn} --attack DTMI --target --MI --TI --DI --saveperts --file_tailor motivation'
baseline_IFGSM_cmd = 'python main.py --gpu {gpu} --white_box {model} --loss_fn {loss_fn} --attack DTMI --target --no-MI --no-TI --no-DI --saveperts --file_tailor motivation_IFGSM'
local_and_fsl_cmd = 'python main.py --gpu {gpu} --white_box {model} --loss_fn {loss_fn} --attack DTMI_Local_FeatureSimilarityLoss --target --MI --TI --DI --saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.4 --depth 3 --file_tailor motivation'


# evaluate the universality
eval_IFGSM_universal_cmd = 'python eval_universal.py --gpu {gpu} --file_tailor Paper-densenet121-NIPSDataset-DTMI-Target_CE-motivation_IFGSM'
eval_DTMI_universal_cmd = 'python eval_universal.py --gpu {gpu} --file_tailor Paper-densenet121-NIPSDataset-DTMI-Target_CE-motivation'

# evaluate similar features
eval_IFGSM_feature_cmd = 'python eval_features.py --gpu {gpu} --file_tailor Paper-densenet121-NIPSDataset-DTMI-Target_CE-motivation_IFGSM'
eval_DTMI_feature_cmd = 'python eval_features.py --gpu {gpu} --file_tailor Paper-densenet121-NIPSDataset-DTMI-Target_CE-motivation'
eval_DTMI_local_sfl_feature_cmd = 'python eval_features.py --gpu {gpu} --file_tailor Paper-densenet121-NIPSDataset-DTMI_Local_FeatureSimilarityLoss-Target_CE-motivation'

if __name__ == '__main__':
    NUM_GPUS = 4
    PROC_PER_GPU = 1
    queue = Queue()
    
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)

    loss_fn = 'CE'
    white_box = 'densenet121'

    # generate adversarial perturbations
    def attack(attack_cmd, model, loss_fn):
        gpu_id = queue.get()
        try: 
            ident = current_process().ident
            print('{}: starting process on GPU {}'.format(ident, gpu_id))
            os.system(attack_cmd.format(gpu=gpu_id, model=model, loss_fn=loss_fn))
        finally:
            queue.put(gpu_id)

    pool = Pool(processes=PROC_PER_GPU * NUM_GPUS)
    attack_cmds = [baseline_DTMI_cmd, baseline_IFGSM_cmd, local_and_fsl_cmd,
            eval_IFGSM_universal_cmd, eval_DTMI_universal_cmd,
            eval_IFGSM_feature_cmd, eval_DTMI_feature_cmd, eval_DTMI_local_sfl_feature_cmd]
    for attack_cmd in attack_cmds:
        pool.apply_async(attack, (attack_cmd, white_box, loss_fn,))

    pool.close()
    pool.join()

  