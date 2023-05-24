import numpy as np

class trainingCurriculum():
    def getWCSTStages(self):
        stages = []
        stages.append({'switchToNextAt': 2000, #8000,
                       'numFeatures': 2,
                       'numFeatureVals': 2,
                       'numObjects': 2,
                       'numLocations': 2,
                       'lrFactor': 1,
                       'ruleSwitchLoopCnt': 16,
                       'hpcNetResetLoopCnt': 48,
                       'trials_per_loop': 10,
                       'ov_per_loop': 5
                       })
        stages.append({'switchToNextAt': 6000, #16000,
                       'numFeatures': 2,
                       'numFeatureVals': 2,
                       'numObjects': 2,
                       'numLocations': 3,
                       'lrFactor': 1,
                       'ruleSwitchLoopCnt': 8,
                       'hpcNetResetLoopCnt': 64,
                       'trials_per_loop': 15,
                       'ov_per_loop': 8
                       })
        stages.append({'switchToNextAt': 8500, #24000,
                       'numFeatures': 2,
                       'numFeatureVals': 2,
                       'numObjects': 2,
                       'numLocations': 4,
                       'lrFactor': 1,
                       'ruleSwitchLoopCnt': 8,
                       'hpcNetResetLoopCnt': 64,
                       'trials_per_loop': 25,
                       'ov_per_loop': 13
                       })
        stages.append({'switchToNextAt': 33000, #60000,
                       'numFeatures': 2,
                       'numFeatureVals': 3,
                       'numObjects': 3,
                       'numLocations': 4,
                       'lrFactor': 0.75,
                       'ruleSwitchLoopCnt': 24,
                       'hpcNetResetLoopCnt': 72,
                       'trials_per_loop': 30,
                       'ov_per_loop': 22
                       })
        stages.append({'switchToNextAt': 43000, #80000,
                       'numFeatures': 2,
                       'numFeatureVals': 3,
                       'numObjects': 3,
                       'numLocations': 4,
                       'lrFactor': 0.75,
                       'ruleSwitchLoopCnt': 16,
                       'hpcNetResetLoopCnt': 90,
                       'trials_per_loop': 30,
                       'ov_per_loop': 15
                       })
        stages.append({'switchToNextAt': 90000, #125000,
                       'numFeatures': 3,
                       'numFeatureVals': 3,
                       'numObjects': 3,
                       'numLocations': 4,
                       'lrFactor': 0.5,
                       'ruleSwitchLoopCnt': 40,
                       'hpcNetResetLoopCnt': 120,
                       'trials_per_loop': 30,
                       'ov_per_loop': 22
                       })
        stages.append({'switchToNextAt': 110000, #175000,
                       'numFeatures': 3,
                       'numFeatureVals': 3,
                       'numObjects': 3,
                       'numLocations': 4,
                       'lrFactor': 0.5,
                       'ruleSwitchLoopCnt': 32,
                       'hpcNetResetLoopCnt': 190,
                       'trials_per_loop': 30,
                       'ov_per_loop': 15
                       })
        stages.append({'switchToNextAt': 125000, #225000,
                       'numFeatures': 3,
                       'numFeatureVals': 3,
                       'numObjects': 3,
                       'numLocations':4,
                       'lrFactor': 0.25,
                       'ruleSwitchLoopCnt': 16,
                       'hpcNetResetLoopCnt': 240,
                       'trials_per_loop': 30,
                       'ov_per_loop': 15
                       })

        return stages

    def getNBackStages(self):
        stages = []
        stages.append({'switchToNextAt': np.inf, #8000,
                       'lrFactor': 1,
                       'hpcNetResetLoopCnt': 2,
                       'trials_per_loop': 2,
                       'ov_per_loop': 0
                       })
        return stages