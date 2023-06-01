import numpy as np

#Base class
class Task:
    def __init__(self):
        pass

    # Generate dataset
    def generateData(self):
        pass

    # Update config parameters (e.g. for new curriculum stage)
    def updateConfigTaskParams(self):
        pass

    # Update control parameters (e.g. for new curriculum stage)
    def updateControlParams(self):
        pass

    # Default config parameters for task
    def getDefaultConfig(self):
        pass

class nBack(Task):
    # Task presents series of alphabet, digit pairs (3 pairs) during sample stim
    # Following a short delay period, model is queried with an alphabet
    # and must respond with corresponding digit

    # Default config parameters for nback
    def getDefaultConfig(self):
        config = {'tau_pfc': 50,  # ms
                  'tau_hpc': 25,  # 100,           # ms
                  'dt': 1,  # discretization time step
                  'sigma_rec': 0.05, # noise width
                  'l2_h': 0.0001, # activity regularization
                  'l2_wRH': 0.5, # weight regularization
                  'seed': 0,
                  'save_name': '3Back',
                  'init_lr_full': 0.001, # learning rate
                  'batch_size': 30,
                  'training_iters': 6000,
                  'trials_per_loop': 2,
                  'ov_per_loop': 0,
                  'hpcNetResetLoopCnt': 2, # frequency of resetting HPC net state
                  'num_hidden_pfc_units': 100,
                  'num_hidden_hpc_units': 100,
                  'num_hidden_eMLP_units': 50,
                  'l_init': 4.60,  #0.99 # HPC wt decay lambda
                  'epsilon': 1e-6,
                  'theta': 0.2, # HPC update parameter for activity baseline
                  'max_gradient_norm': 1.0,
                  'sigma': 0.15, # HPC update parameter for currents scaling
                  'ckpt_dir': 'checkpoints3Back',
                  'save_every': 500,  # save every 500 rule blocks
                  'save_data': False
                  }

        config['alpha_pfc'] = 1.0 * config['dt'] / config['tau_pfc']
        config['alpha_hpc'] = 1.0 * config['dt'] / config['tau_hpc']

        return config

    # Update config parameters
    def updateConfigTaskParams(self, config, stages):

        # Stim parameters
        config['numClasses'] = 26 + 10 + 1
        config['numObjects'] = 9
        config['num_input'] = config['numClasses']
        config['num_samp_stim'] = config['numClasses']

        config['num_output'] = 10  # objects

        # Trial duration parameters
        config['stim_dur'] = int(100 / config['dt'])
        config['stimPeriod'] = config['numObjects'] * np.array([0, config['stim_dur']])  # also fixate here
        config['time_to_resp'] = config['stimPeriod'][1]
        config['resp_dur'] = int(200 / config['dt'])
        config['responsePeriod'] = config['time_to_resp'] + np.array([0, config['resp_dur']])
        config['time_to_iti'] = config['responsePeriod'][1]
        config['ITI_dur'] = int(200.0 / config['dt'])
        config['ITIPeriod'] = config['time_to_iti'] + np.array([0, config['ITI_dur']])
        config['tdim'] = config['ITIPeriod'][1]
        config['outcomeFeedback'] = False # no feedback in nback task

        return config

    # Update control parameters
    def updateControlParams(self, stage, FLAGS):
        FLAGS.curr_trials_per_loop = stage['trials_per_loop']
        FLAGS.curr_ov_per_loop = stage['ov_per_loop']
        FLAGS.hpcNetResetLoopCnt = stage['hpcNetResetLoopCnt']
        return FLAGS

    # Generate nback dataset
    def generateData(self, FLAGS, trial, currentRule, stage):
        # target, stims and loss mask for each trial in each sample in minibatch
        targetList = np.zeros((FLAGS.batch_size, FLAGS.num_output, FLAGS.curr_trials_per_loop))
        allStims = np.zeros((FLAGS.batch_size, FLAGS.numObjects, FLAGS.curr_trials_per_loop, FLAGS.num_input))
        lossMask = np.zeros((FLAGS.batch_size, FLAGS.trials_per_loop))

        # loop through minibatch sample
        for b in range(FLAGS.batch_size):
            lossMask[b,:] = 1 # no loss masking
            ii = np.random.choice(26, 1) # which alphabet to query
            ls = np.concatenate((np.arange(0, ii), np.arange(ii + 1, 26))) # other alphabets
            ns = np.arange(26, 36) # digits

            for t in range(FLAGS.curr_trials_per_loop): # loop through trials in sample
                pick = np.random.choice(ls, 2, replace=False) # select other alphabets
                pickn = np.random.choice(ns, 3, replace=False) # select digits
                # allStims[b,[0,2],t,pick[0:2]] = 1
                # allStims[b,4,t,ii] = 1
                allStims[b,[2,4],t,pick[0:2]] = 1
                allStims[b,0,t,ii] = 1
                allStims[b,[1,3,5],t,pickn[0:3]] = 1
                allStims[b,[6,7],t,36] = 1
                allStims[b,8,t,ii] = 1
                # targetList[b,pickn[2]-26,t] = 1
                targetList[b,pickn[0]-26,t] = 1

        return allStims.astype(np.float32), targetList.astype(np.float32), currentRule, lossMask.astype(np.float32)


class WCST(Task):
    # Task presents k objects each with a distinct feature (e.g. square, triangle, etc.)
    # from each possible dimension (e.g. shape)
    # One of the features is the hidden rule and must be inferred by trial and error (based on feedback)
    # Hidden rule determines which object must be selected

    # Default config parameters for WCST analog
    def getDefaultConfig(self):

        # Set default configuration parameters
        config = {'tau_pfc': 100,  # ms
                  'tau_hpc': 50,  # 100,           # ms
                  'dt': 5,  # discretization time step
                  'sigma_rec': 0.05, # noise width
                  'l2_h': 0.0001, # activity regularization
                  'l2_wRH': 0.5, # weight regularization
                  'seed': 0,
                  'save_name': 'WCST',
                  'init_lr_full': 0.001,  # 0.0001 # learning rate
                  'batch_size': 72,
                  'training_iters': 250000,
                  'trials_per_loop': 30,
                  'ov_per_loop': 22,
                  'ruleSwitchLoopCnt': 0, # frequency of switching rule
                  'hpcNetResetLoopCnt': 0, # frequency of resetting HPC net state
                  'num_hidden_pfc_units': 100,
                  'num_hidden_hpc_units': 100,
                  'num_hidden_eMLP_units': 50,
                  'l_init': 3.60,  # 4.60 #0.99 # HPC wt decay lambda
                  'epsilon': 1e-6,
                  'theta': 0.2, # HPC update parameter for activity baseline
                  'max_gradient_norm': 0.5,
                  'sigma': 0.15, # HPC update parameter for currents scaling
                  'ckpt_dir': 'checkpointsWCST',  # 'err',
                  'save_every': 500,  # save every 500 rule blocks
                  'save_data': False
                  }

        config['alpha_pfc'] = 1.0 * config['dt'] / config['tau_pfc']
        config['alpha_hpc'] = 1.0 * config['dt'] / config['tau_hpc']

        return config

    # Update config parameters (e.g. for new curriculum stage)
    def updateConfigTaskParams(self, config, stages):

        # Stim parameters
        config['numFeatures'] = np.max([stage['numFeatures'] for stage in stages]) # no. of dimensions
        config['numFeaturesVals'] = np.max([stage['numFeatureVals'] for stage in stages]) # no. of features per dim.
        config['numObjects'] = max(np.max([stage['numObjects'] for stage in stages]),
                                   np.max([stage['numLocations'] for stage in stages]))
        config['num_samp_stim'] = config['numFeatures'] * config['numFeaturesVals']
        config['num_input'] = config['num_samp_stim'] + 1 + config['numObjects']  # objects*dimensions/features*feature identities + Reward + action selection

        config['num_output'] = config['numObjects']  # objects

        # Trial duration parameters
        config['stim_dur'] = int(100 / config['dt'])
        config['stimPeriod'] = config['numObjects'] * np.array([0, config['stim_dur']])  # also fixate here
        config['time_to_resp'] = config['stimPeriod'][1]
        config['responsePeriod'] = config['time_to_resp'] + np.array([0, int(200 / config['dt'])])
        config['time_to_iti'] = config['responsePeriod'][1]
        config['ITIPeriod'] = config['time_to_iti'] + np.array([0, int(300 / config['dt'])])
        config['tdim'] = config['ITIPeriod'][1]
        config['outcomeFeedback'] = True

        return config

    # Update control parameters (e.g. for new curriculum stage)
    def updateControlParams(self, stage, FLAGS):
        FLAGS.curr_trials_per_loop = stage['trials_per_loop']
        FLAGS.curr_ov_per_loop = stage['ov_per_loop']
        FLAGS.hpcNetResetLoopCnt = stage['hpcNetResetLoopCnt']
        FLAGS.ruleSwitchLoopCnt = stage['ruleSwitchLoopCnt']
        return FLAGS

    # Generate WCST analog dataset
    def generateData(self, FLAGS, trial, currentRule, stage):
        # generate possible rules (stage-dependent)
        numRules = stage['numFeatures'] * stage['numFeatureVals']
        ruleRange = [i + j*FLAGS.numFeaturesVals for i in range(stage['numFeatureVals']) for j in range(stage['numFeatures'])]

        if (len(currentRule) == 0) or (trial % FLAGS.ruleSwitchLoopCnt == 0):
            currentRule = np.zeros((FLAGS.batch_size,1),dtype=int)

            # select first rule from possible rules
            o = np.random.permutation(ruleRange)
            pre = 0
            for i in range(len(o)):
                currentRule[pre:((i+1)*(FLAGS.batch_size//numRules)),0] = o[i]
                pre = (i+1)*(FLAGS.batch_size//numRules)

        # Assume num objects = num features vals
        targetList = np.zeros((FLAGS.batch_size, FLAGS.num_output, FLAGS.trials_per_loop))
        allStims = np.zeros((FLAGS.batch_size, FLAGS.numObjects, FLAGS.trials_per_loop, FLAGS.num_samp_stim))
        lossMask = np.zeros((FLAGS.batch_size, FLAGS.trials_per_loop))
        for j in range(FLAGS.trials_per_loop): # generate for each trial in sample
            if j < FLAGS.curr_trials_per_loop: # curriculum might require training on fewer trials per sample
                lossMask[:,j] = 1.0
            # stims will have shape batch x objects x features*feature vals
            fmat = []
            # discrete spatial location of objects
            objLocations = np.array([np.random.choice(np.arange(stage['numLocations']), stage['numFeatureVals'], replace=False) for i in range(FLAGS.batch_size)])
            # generate objects as random combinations of 1 feature val per feature
            for i in range(stage['numFeatures']):
                ff = np.tile(np.arange(stage['numFeatureVals']),(FLAGS.batch_size,1))
                idx = np.random.rand(*ff.shape).argsort(axis=1)
                fs = np.take_along_axis(ff, idx, axis=1)
                fmat.append(fs + i*FLAGS.numFeaturesVals)
                fs = (np.arange(fs.max() + 1) == fs[..., None]).astype(np.float32)

                # allStims[:, 0:stage['numFeatureVals'], j, (i * FLAGS.numFeaturesVals):((i* FLAGS.numFeaturesVals)+stage['numFeatureVals'])] = fs
                allStims[np.reshape(np.arange(FLAGS.batch_size),(FLAGS.batch_size,1)), objLocations, j, (i * FLAGS.numFeaturesVals):((i* FLAGS.numFeaturesVals)+stage['numFeatureVals'])] = fs

            # generate target based on rule
            fmat = np.stack(fmat, axis = 2)
            for rule in np.unique(currentRule[:,0]):
                bs = np.where(currentRule[:,0] == rule)[0]

                inds = np.where(fmat[bs,:,:] == rule)
                # targetList[bs[inds[0]], inds[1],j] = 1
                targetList[bs[inds[0]], objLocations[bs[inds[0]], inds[1]] ,j] = 1

        allStims = allStims[:,:,0:FLAGS.curr_trials_per_loop,:]
        targetList = targetList[:,:,0:FLAGS.curr_trials_per_loop]
        lossMask = lossMask[:,0:FLAGS.curr_trials_per_loop]

        return allStims.astype(np.float32), targetList.astype(np.float32), currentRule, lossMask.astype(np.float32)
