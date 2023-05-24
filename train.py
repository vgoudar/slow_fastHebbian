import os
import time
import numpy as np
# import math
import pickle
import torch

from stages import *
from network import setupTraining, runMiniBatch
from tasks import *

class Dict2Class(object):

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def restoreModel(PATH, model, optimizer):
    checkpoint = torch.load(PATH)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

# Generate model and train network
def train(seed = 0, chlModInhOff=0.3, hpcEncLag=1, restore = False, task='WCST'):
    stages = None
    if task == 'WCST':
        task = WCST()
        curriculum = trainingCurriculum()
        stages = curriculum.getWCSTStages()
    elif task == '3Back':
        task = nBack()
        curriculum = trainingCurriculum()
        stages = curriculum.getNBackStages()
    else:
        raise Exception("Not implemented")


    # Setup hyper-parameters
    config = task.getDefaultConfig()
    config['ckpt_dir'] = config['ckpt_dir'] + '/' + str(seed)
    config['seed']          = seed
    save_name = '{:s}_{:d}_{:d}_{:f}'.format(config['save_name'], seed, hpcEncLag, chlModInhOff)
    config['save_name']     = save_name

    config = task.updateConfigTaskParams(config, stages)

    config['chlModInhOff'] = chlModInhOff
    config['hpcEncLag'] = hpcEncLag
    config['saveLag'] = config['hpcEncLag']
    if config['saveLag'] < 1:
        config['saveLag'] = 1

    # Display configuration
    for key, val in config.items():
        print('{:20s} = '.format(key) + str(val))

    FLAGS = Dict2Class(config)

    if not os.path.isdir(FLAGS.ckpt_dir):
        os.makedirs(FLAGS.ckpt_dir)

    with open(os.path.join(FLAGS.ckpt_dir, "%s_conf.pkl" % FLAGS.save_name), 'wb') as f:
        pickle.dump(FLAGS, f)
        pickle.dump(stages, f)

    initStates0 = np.zeros([FLAGS.batch_size, config['saveLag'], FLAGS.num_hidden_hpc_units], dtype=np.float32)
    consecutiveCorrect0 = np.zeros([FLAGS.batch_size, 1], dtype=np.float32)
    A_init0 = np.zeros([FLAGS.batch_size, FLAGS.num_hidden_hpc_units, FLAGS.num_hidden_hpc_units], dtype=np.float32)

    model, lClass, optimizer = setupTraining(FLAGS)
    perfA = dict()
    if restore:
        npzfile = np.load(FLAGS.ckpt_dir + '/contInp_' + FLAGS.save_name + '.npz')
        varNames = npzfile.files[0:12]
        print(varNames, flush = True)
        ldA_init = npzfile[varNames[0]]
        ldInitStates = npzfile[varNames[1]]
        ldConsecutiveCorrect = npzfile[varNames[2]]
        ldCurrentRule = npzfile[varNames[3]]
        
        with open(FLAGS.ckpt_dir + '/contInp_' + FLAGS.save_name + '.out', "rb") as fp:
           b = pickle.load(fp)
           print(b, flush = True)
           ldTrial = b[0]
           ldCurrTrial = b[1]
           ldStageid = b[2]
           ldGlobal_step = b[3]

        PATH = os.path.join(FLAGS.ckpt_dir, "%s_%d_%d_check.pt" % (FLAGS.save_name, ldGlobal_step, ldStageid))

        model, optimizer = restoreModel(PATH, model, optimizer)

        npzfile = np.load(os.path.join(FLAGS.ckpt_dir, "%s_perf.npz" % FLAGS.save_name))
        perfA['loss'] = npzfile['perfALoss']
        perfA['rewards'] = npzfile['perfARewards']
        perfA['accuracy'] = npzfile['perfAAccuracy']
    else:
        perfA['loss'] = []
        perfA['rewards'] = []
        perfA['accuracy'] = []

    model.train(True)

    t_start = time.time()

    if FLAGS.save_data:
        RHHA = None
        RHPA = None
        EVA = None
        REWA = None
        CRA = None

    perf = dict()
    perf['loss'] = []
    perf['rewards'] = []
    perf['accuracy'] = []

    if restore:
        currentRule = ldCurrentRule
        stageid = ldStageid
        currTrial = ldCurrTrial
        global_step = ldGlobal_step
        trialInit = ldTrial
        A_init = ldA_init
        initStates = ldInitStates
        consecutiveCorrect = ldConsecutiveCorrect
    else:
        currentRule = []
        stageid = 0
        global_step = 0
        trialInit = 0

    FLAGS = task.updateControlParams(stages[stageid], FLAGS)

    if restore == False:
        currTrial = -1*(FLAGS.curr_trials_per_loop - FLAGS.curr_ov_per_loop)

    for trial in range(trialInit, FLAGS.training_iters):
        currTrial = currTrial + (FLAGS.curr_trials_per_loop - FLAGS.curr_ov_per_loop)
        changeLR = None

        if (currTrial % FLAGS.hpcNetResetLoopCnt) == 0:
            A_init = A_init0
            initStates = initStates0
            if trial >= stages[stageid]['switchToNextAt']:  # 10000
                stageid = stageid + 1
                changeLR = stages[stageid]['lrFactor']
                FLAGS.hpcNetResetLoopCnt = stages[stageid]['hpcNetResetLoopCnt']
                FLAGS.ruleSwitchLoopCnt = stages[stageid]['ruleSwitchLoopCnt']
                FLAGS.curr_trials_per_loop = stages[stageid]['trials_per_loop']
                FLAGS.curr_ov_per_loop = stages[stageid]['ov_per_loop']
                consecutiveCorrect = consecutiveCorrect0
                currentRule = []

                global_step = global_step + 1

                PATH = os.path.join(FLAGS.ckpt_dir, "%s_%d_%d_check.pt" % (FLAGS.save_name, global_step-1, stageid-1))
                print("Saving the model before stage switch.")
                torch.save({
                    'epoch': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, PATH)                                
                print('stageid set to ' + str(stageid), flush=True)
                with open(FLAGS.ckpt_dir + '/contInp_' + FLAGS.save_name + '.out', "wb") as fp:   #Pickling
                    pickle.dump([trial, currTrial-(FLAGS.curr_trials_per_loop - FLAGS.curr_ov_per_loop), stageid-1, global_step-1], fp)
                arrLst= [A_init, initStates, consecutiveCorrect, currentRule]
                np.savez(FLAGS.ckpt_dir + '/contInp_' + FLAGS.save_name + '.npz' , *arrLst)


        if hasattr(FLAGS, 'ruleSwitchLoopCnt') and (currTrial % FLAGS.ruleSwitchLoopCnt) == 0:
            consecutiveCorrect = consecutiveCorrect0

        # Generate a batch of trials
        gdStartTime = time.time()
        stimList, targetList, currentRule, lossMask = task.generateData(FLAGS, currTrial, currentRule, stages[stageid])
        gdEndTime = time.time()


        tfStartTime = time.time()
        loss, cost_reg, accuracy, rewards, A_init, initStates, consecutiveCorrect, dec, e_bias, savedData = \
            runMiniBatch(model, stimList, targetList,
                         lossMask, A_init, initStates,
                         consecutiveCorrect, FLAGS.curr_trials_per_loop-FLAGS.curr_ov_per_loop,
                         lClass, optimizer, FLAGS.curr_trials_per_loop, changeLR=changeLR)
        tfEndTime = time.time()

        consecutiveCorrect = consecutiveCorrect[:, :, -1]
        if FLAGS.save_data:
            print(
                'Training: %d, Loss: %f (%f), Accuracy: %f (%f, %f), Norms: %f / %f, gNorm: %f, decay: %f, e_bias: %f, e_vals: (%f, %f, %f, %f)' %
                (trial, loss, cost_reg, np.mean(rewards), accuracy, np.mean(consecutiveCorrect), savedData['rhNorms'], savedData['rpNorms'], savedData['gnorm'],
                 dec, e_bias,
                 np.max(savedData['eHist']), np.min(savedData['eHist']), np.mean(savedData['eHist']), np.median(savedData['eHist'])), flush=True)
        else:

            print('Training: %d, Loss: %f (%f), Accuracy: %f (%f, %f), decay: %f, e_bias: %f' %
                (trial, loss, cost_reg, np.mean(rewards), accuracy, np.mean(consecutiveCorrect), dec, e_bias), flush=True)


        # if math.isnan(loss):
        #     raise Exception('It has happened with nanloss')

        if FLAGS.save_data:
            if RHHA is None:
                RHHA = savedData['RH_H']
                RHPA = savedData['RH_P']
                EVA = np.squeeze(savedData['eHist']).T
                REWA = np.squeeze(rewards)
                CRA = currentRule
            else:
                RHHA = np.concatenate((RHHA, savedData['RH_H']), axis=2)
                RHPA = np.concatenate((RHPA, savedData['RH_P']), axis=2)
                EVA = np.concatenate((EVA, np.squeeze(savedData['eHist']).T), axis=1)
                REWA = np.concatenate((REWA, np.squeeze(rewards)), axis=1)
                if not currentRule:
                    CRA = np.array(currentRule)
                else:
                    CRA = np.concatenate((CRA, currentRule), axis=1)

        perf['loss'] = []
        perf['rewards'] = []
        perf['accuracy'] = []

        perf['loss'] .append(loss)
        perf['rewards'].append(np.mean(rewards))
        perf['accuracy'].append(accuracy)
        runTime = time.time()-t_start
        if trial%100 == 0:
            print('Trial: ' + str(trial) + ' loss: ' + str(np.mean(perf['loss'])) + ' rewards: ' + str(np.mean(perf['rewards'])) + ' accuracy: ' + str(np.mean(perf['accuracy'])) + ' Runtime: ' + str(runTime) + ' s (' + str(gdEndTime-gdStartTime) + ', ' + str(tfEndTime-tfStartTime) + ')')
            # print(str(perf['loss'] [-10:]))
            # print(str(perf['rewards'][-10:]))
            # print(str(perf['accuracy'][-10:]))
            perfA['loss'].extend(perf['loss'])
            perfA['rewards'].extend(perf['rewards'])
            perfA['accuracy'].extend(perf['accuracy'])
            perf['loss'] = []
            perf['rewards'] = []
            perf['accuracy'] = []

        #if math.isnan(loss):
        #    exit()

        if (trial % FLAGS.save_every == 0) and \
                (trial > 0):
            PATH = os.path.join(FLAGS.ckpt_dir, "%s_%d_%d_check.pt" % (FLAGS.save_name, global_step, stageid))
            print("Saving the model.")
            torch.save({
                'epoch': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, PATH)
            with open(FLAGS.ckpt_dir + '/contInp_' + FLAGS.save_name + '.out', "wb") as fp:   #Pickling
                pickle.dump([trial+1, currTrial, stageid, global_step], fp)
            arrLst= [A_init, initStates, consecutiveCorrect, currentRule]
            np.savez(FLAGS.ckpt_dir + '/contInp_' + FLAGS.save_name + '.npz' , *arrLst)

            np.savez(os.path.join(FLAGS.ckpt_dir, "%s_perf.npz" % FLAGS.save_name), perfALoss=perfA['loss'], perfARewards=perfA['rewards'], perfAAccuracy=perfA['accuracy'])


if __name__ == '__main__':
    train(seed = 3, chlModInhOff=0.3, hpcEncLag=1)

