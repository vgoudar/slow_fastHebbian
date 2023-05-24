import numpy as np
import torch
from torch import nn
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNNCell(nn.RNNCell):  # Euler integration of rate-neuron network dynamics
    def __init__(self, input_size, hidden_size, h2h_size, alpha, sigma, h2h_initStd=0.01, bias=True):
        super().__init__(input_size, hidden_size, bias, nonlinearity='relu')

        torch.nn.init.orthogonal_(self.weight_ih, gain=2.0)

        torch.nn.init.eye_(self.weight_hh)

        updW = (self.weight_hh * 0.8).clone().detach()
        self.weight_hh = torch.nn.Parameter(updW)

        torch.nn.init.zeros_(self.bias_hh)

        self.weight_h2h = torch.nn.Parameter(torch.empty((h2h_size, hidden_size)))
        torch.nn.init.normal_(self.weight_h2h, mean=0.0, std=h2h_initStd)

        self.alpha = np.float32(alpha)
        self.sigma = np.float32(sigma)
        self.nonlinearity_function = nn.ReLU()

    def forward(self, input, hidden, hidden_other):
        activity = input @ self.weight_ih.t() + hidden @ self.weight_hh.t() + hidden_other @ self.weight_h2h.t() + self.bias_hh
        activity = activity + hidden.data.new(hidden.size()).normal_(0, self.sigma)
        
        activity = self.nonlinearity_function(activity)
        hidden = (1.0 - self.alpha) * hidden + self.alpha * activity
        return hidden

class RNNCellFastWts(nn.RNNCell):  # Euler integration of rate-neuron network dynamics with dynamic hebbian updates
    def __init__(self, input_size, hidden_size, h2h_size, alpha, sigma, sigma_non_lin, theta_non_lin, l_init, epsilon, dt, h2h_initStd=0.005, hpcEncLag = 1, bias=False):
        super().__init__(input_size, hidden_size, bias, nonlinearity='relu')

        torch.nn.init.orthogonal_(self.weight_ih, gain=2.0)

        self.weight_h2h = torch.nn.Parameter(torch.empty((h2h_size, hidden_size)))
        torch.nn.init.normal_(self.weight_h2h, mean=0.0, std=h2h_initStd)

        self.fastWts = None

        self.l = torch.nn.Parameter(torch.empty((1)))
        torch.nn.init.constant_(self.l, np.float32(l_init))

        self.alpha = np.float32(alpha)
        self.sigma = np.float32(sigma)
        self.sigma_non_lin = np.float32(1.0 / (np.sqrt(2.0) * sigma_non_lin))
        self.theta_non_lin = np.float32(theta_non_lin)
        self.sc = np.float32(5.0/hidden_size)
        self.nonlinearity_function = nn.ReLU()
        self.register_buffer('non_lin_min', torch.as_tensor(np.float32(10.0)))
        self.hpcEncLag = hpcEncLag
        self.epsilon = np.float32(epsilon)
        self.dt = dt
        self.register_buffer('neg_inv_dt', torch.as_tensor(np.float32(-1.0 / self.dt)))
        self.register_buffer('l_min', torch.as_tensor(np.float32(6.0)))

    def forward(self, input, hidden, hidden_other):
        activity = torch.mul(self.fastWts @ torch.unsqueeze(hidden, 2), self.sc)
        activity = torch.squeeze(activity, 2) + input @ self.weight_ih.t() + hidden_other @ self.weight_h2h.t()
        activity = activity + hidden.data.new(hidden.size()).normal_(0, self.sigma)
        activity = torch.minimum(self.nonlinearity_function((activity - self.theta_non_lin) * self.sigma_non_lin), self.non_lin_min)
        hidden = (1.0 - self.alpha) * hidden + self.alpha * activity
        return hidden

    def resetFastWts(self, fastWts0, device):
        with torch.no_grad():
            self.fastWts = torch.as_tensor(fastWts0).to(device)

    def setHiddenInit(self, hidden_init, device):
        with torch.no_grad():
            self.hidden_init = torch.as_tensor(hidden_init).to(device)

    def standardize(self, tensor):
        mean = torch.mean(tensor, dim = 1, keepdim=True)
        var = torch.var(tensor, dim = 1, unbiased = False, keepdim=True) + self.epsilon
        std = torch.sqrt(var)
        standardized = torch.div((tensor-mean), std)
        return standardized

    def updateFastWts(self, hidden, hidden_old, t, e):
        post = self.standardize(hidden)
        if self.hpcEncLag == 0:
            pre = self.standardize(hidden)
        elif t < self.hpcEncLag:
            pre = self.standardize(self.hidden_init[:, t - self.hpcEncLag, :])
        else:
            pre = self.standardize(hidden_old[t - self.hpcEncLag])

        tmp = torch.squeeze(self.fastWts @ torch.unsqueeze(pre, 2), 2)

        inc = torch.einsum('bi,bj->bij', post - tmp, pre)
        self.fastWts = self.fastWts + self.dt * torch.add(torch.mul(
            torch.maximum(-1.0 * torch.sigmoid(-1.0 * torch.minimum(self.l, self.l_min)), self.neg_inv_dt),
            self.fastWts),
            torch.einsum('b,bjk->bjk', e, inc))


class pfc_hpc_WCST_model(nn.Module):
    def __init__(self, FLAGS):
        print('Using device:', device, flush=True)

        super(pfc_hpc_WCST_model, self).__init__()

        sigma_pfc = np.sqrt(2 * FLAGS.alpha_pfc) * FLAGS.sigma_rec
        sigma_hpc = np.sqrt(2 * FLAGS.alpha_hpc) * FLAGS.sigma_rec

        self.pfc = RNNCell(FLAGS.num_input,
                           FLAGS.num_hidden_pfc_units,
                           FLAGS.num_hidden_hpc_units,
                           FLAGS.alpha_pfc,
                           sigma_pfc)

        self.eMLP = nn.Linear(FLAGS.num_hidden_pfc_units, FLAGS.num_hidden_eMLP_units)
        torch.nn.init.normal_(self.eMLP.weight, mean=0.0, std=0.005)
        torch.nn.init.zeros_(self.eMLP.bias)
        self.eMLP_nl = nn.ReLU()

        self.pe = nn.Linear(FLAGS.num_hidden_eMLP_units, 1)
        torch.nn.init.normal_(self.pe.weight, mean=0.0, std=0.01)
        updW = (self.pe.weight - np.float32(FLAGS.chlModInhOff)).clone().detach()
        self.pe.weight = torch.nn.Parameter(updW)
        torch.nn.init.constant_(self.pe.bias, np.float32(- 10.0))
        self.pe_nl = nn.Sigmoid()


        self.hpc = RNNCellFastWts(FLAGS.num_input,
                                  FLAGS.num_hidden_hpc_units,
                                  FLAGS.num_hidden_pfc_units,
                                  FLAGS.alpha_hpc,
                                  sigma_hpc,
                                  FLAGS.sigma,
                                  FLAGS.theta,
                                  FLAGS.l_init,
                                  FLAGS.epsilon,
                                  FLAGS.dt,
                                  hpcEncLag = FLAGS.hpcEncLag)

        self.output = nn.Linear(FLAGS.num_hidden_pfc_units, FLAGS.num_output)
        torch.nn.init.uniform_(self.output.weight, a=-np.sqrt(2.0 / FLAGS.num_hidden_pfc_units), b=np.sqrt(2.0 / FLAGS.num_hidden_pfc_units))
        torch.nn.init.zeros_(self.output.bias)
        self.o_nl = nn.Softmax(dim=1)


        self.pfc.to(device)
        self.eMLP.to(device)
        self.pe.to(device)
        self.hpc.to(device)
        self.output.to(device)

        self.device = device
        self.FLAGS = FLAGS
                
        self.register_buffer('l2_wRH', torch.as_tensor(np.float32(FLAGS.l2_wRH)))
        self.register_buffer('l2_h', torch.as_tensor(np.float32(FLAGS.l2_h)))
        

    def init_pass(self, batch_size, A_init, rh_init, cc, X, y, endov):
        rp = torch.zeros(batch_size, self.FLAGS.num_hidden_pfc_units).to(self.device)
        rh = torch.as_tensor(rh_init[:,-1,:]).to(self.device)
        self.hpc.resetFastWts(A_init, self.device)
        self.hpc.setHiddenInit(rh_init, self.device)
        consecutiveCorrect = torch.as_tensor(cc).to(self.device)
        zeroStim = torch.zeros(batch_size, self.FLAGS.num_tot_visf).to(self.device)
        X  = torch.as_tensor(X).to(self.device)
        y = torch.as_tensor(y).to(self.device)
        endov = torch.as_tensor(endov).to(self.device)
        reward = torch.zeros(batch_size, 1).to(self.device)
        actionSel = torch.zeros(batch_size, self.FLAGS.numObjects).to(self.device)

        return rp, rh, consecutiveCorrect, reward, actionSel, zeroStim, X, y, endov

    def forward(self, X, y, A_init, rh_init, cc, endov, ntrials):
        batch_size = X.shape[0]
        rp0, rh, consecutiveCorrect, reward0, actionSel0, zeroStim, X, y, endov = self.init_pass(batch_size, A_init, rh_init, cc, X, y, endov)

        gOut = []
        eH = []
        ccs = []
        rHistP_H = []
        rHistP_P = []
        rhNorms = []
        rpNorms = []
        rewHist = []

        for b in range(0, ntrials):

            reward = reward0
            actionSel = actionSel0
            with torch.no_grad():
                target = y[:, :, b]

            hHist = []
            hHist2 = []
            sInd = -1
            rp = rp0
            for t in range(0, self.FLAGS.tdim):
                if t % self.FLAGS.stim_dur == 0 and sInd < self.FLAGS.numObjects:
                    sInd += 1

                if sInd < self.FLAGS.numObjects:
                    stim = X[:, sInd, b, :]
                else:
                    stim = zeroStim

                with torch.no_grad():
                    if self.FLAGS.outcomeFeedback:
                        inp = torch.cat((stim, reward, actionSel), dim=1)
                    else:
                        inp = stim

                rh_new = self.hpc(inp, rh, rp)
                rp = self.pfc(inp, rp, rh)

                trans = self.eMLP_nl(self.eMLP(rp))
                e = np.float32(1.0 / 100.0) * self.pe_nl(self.pe(trans))
                if self.FLAGS.save_data:
                    eH.append(e)

                self.hpc.updateFastWts(rh_new, rHistP_H, t, torch.squeeze(e, dim=1)) #torch.squeeze(e, dim=1)

                rh = rh_new

                if self.FLAGS.save_data:
                    rhNorms.append(torch.norm(rh, p=2, dim = 1))
                    rpNorms.append(torch.norm(rp, p=2, dim = 1))

                rHistP_P.append(rp)
                rHistP_H.append(rh)

                # compute and outputs and target
                if self.FLAGS.time_to_resp <= t < self.FLAGS.time_to_iti:
                    hHist.append(self.output(rp))
                    hHist2.append(self.o_nl(hHist[-1]))


                if t == (self.FLAGS.time_to_iti-1):
                    choice = torch.mean(torch.stack(hHist2, dim=2), dim=2)
                    actInd = torch.squeeze(torch.multinomial(choice, 1))
                    actionSel = torch.nn.functional.one_hot(actInd, self.FLAGS.numObjects).type(torch.float)
                    correctChoice = torch.sum(torch.multiply(actionSel, target), dim=1, keepdim=True)
                    reward = -1.0 + 2.0 * torch.gt(correctChoice, 0.9).type(torch.float)
                    rewHist.append(reward)

                    consecutiveCorrect = torch.multiply(consecutiveCorrect, reward) + reward
                    ccs.append(consecutiveCorrect)

            gOut.append(torch.mean(torch.stack(hHist, dim=2), dim=2))

            if endov == (b+1):
                A_int = torch.clone(self.hpc.fastWts)
                rh_int = torch.stack(rHistP_H[(-self.FLAGS.saveLag):], dim = 1)
            # self.A_int = tf.cond(tf.equal((tf.stop_gradient(self.endov) - 1), b), lambda: tf.identity(self.hpc.fastWts), lambda: self.A_int)
            # self.rh_int = tf.cond(tf.equal((tf.stop_gradient(self.endov) - 1), b), lambda: tf.stack(rHistP_H[(-self.FLAGS.saveLag):], axis=1), lambda: self.rh_int)

        RH_P = torch.stack(rHistP_P, dim=2)
        RH_H = torch.stack(rHistP_H, dim=2)

        # Loss
        logits = torch.stack(gOut, dim=2)
        rewards = torch.stack(rewHist, dim=2)
        ccs = torch.stack(ccs, dim=2)
        if self.FLAGS.save_data:
            savedData = {}
            savedData['eHist'] = torch.stack(eH, dim=0)  # *50.0
            savedData['rhNorms'] = torch.mean(torch.stack(rhNorms, dim=1))
            savedData['rpNorms'] = torch.mean(torch.stack(rpNorms, dim=1))
        else:
            savedData = None

        return logits, rewards, y, RH_P, RH_H, self.hpc.fastWts, A_int, rh_int, ccs, savedData


def setupTraining(FLAGS):
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    model = pfc_hpc_WCST_model(FLAGS)
    for p in model.parameters():
        print(p.shape, flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.init_lr_full)
    lClass = torch.nn.CrossEntropyLoss(reduction='none')

    return model, lClass, optimizer


def runMiniBatch(model, X, y, lossmask, A_init, rh_init, cc, endov, lClass, optimizer, ntrials, changeLR = None):
    if changeLR is not None:
        for g in optimizer.param_groups:
            g['lr'] = changeLR * model.FLAGS.init_lr_full

    optimizer.zero_grad()

    # forward pass
    lossmask = torch.as_tensor(lossmask).to(model.device)
    logits, rewards, y, RH_P, RH_H, A, A_int, rh_int, ccs, savedData = model(X, y, A_init, rh_init, cc, endov, ntrials)
    e_bias = model.pe.bias.item()
    decay = model.hpc.l.item()

    accuracy = torch.mean(torch.eq(torch.argmax(logits, dim=1), torch.argmax(y, dim=1)).type(torch.float))
    # v1 = torch.argmax(logits, dim=1)
    # v2 = torch.argmax(y, dim=1)
    # v3 = torch.eq(v1, v2).type(torch.float)
    # accuracy = torch.mean(v3)

    # compute loss
    v4 = lClass(logits, torch.argmax(y, dim=1))
    loss = torch.mean(torch.multiply(lossmask, v4))

    loss_reg = torch.tensor(0.).to(device)
    if model.FLAGS.l2_h > 0:
        hhNorm = torch.mean(torch.square(RH_H))
        hpNorm = torch.mean(torch.square(RH_P))
        loss_reg += (hhNorm + hpNorm) * model.l2_h

    if model.FLAGS.l2_wRH > 0:
        wNormA = torch.mean(torch.sum(torch.sum(torch.square(A), dim=2), dim=1))
        loss_reg += model.l2_wRH * wNormA

    loss_full = loss + loss_reg

    #backward pass
    loss_full.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=model.FLAGS.max_gradient_norm)
    # print(gnorm, flush=True)
    optimizer.step()

    #convert all to numpy before returning
    cost = loss.item()
    cost_reg = loss_reg.item()
    accuracy = accuracy.item()
    A_int = np.squeeze(A_int.detach().cpu().numpy())
    rh_int = rh_int.detach().cpu().numpy()
    ccs = ccs.detach().cpu().numpy()
    rewards = np.squeeze(rewards.detach().cpu().numpy())

    if model.FLAGS.save_data:
        savedData['eHist'] = np.squeeze(savedData['eHist'].detach().cpu().numpy())
        savedData['rhNorms'] = savedData['rhNorms'].item()
        savedData['rpNorms'] = savedData['rpNorms'].item()
        savedData['gnorm'] = gnorm.item()
        savedData['RH_P'] = np.squeeze(RH_P.detach().cpu().numpy())
        savedData['RH_H'] = np.squeeze(RH_H.detach().cpu().numpy())

    return cost, cost_reg, accuracy, rewards, A_int, rh_int, ccs, decay, e_bias, savedData

