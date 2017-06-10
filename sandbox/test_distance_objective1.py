import numpy as np
import theano
import theano.tensor as T



def theano_test_fion(feat, targets):
    epsilon = 1e-9
    df = T.mean(abs(feat.dimshuffle(['x',0,1]) - feat.dimshuffle([0,'x',1])), axis=2)
    b_targets = (targets > 0.5).nonzero()
    b_no_targets = (targets < 0.5).nonzero()
    df_pp = df[b_targets]
    df_pp = df_pp[:,b_targets]

    df_pn = df[b_targets]
    df_pn = df[:,b_no_targets]

    n_pos = T.cast(T.sum(targets), 'float32')
    n_neg = T.cast(T.sum(1-targets), 'float32')

    avg_dist_pp = T.sum(df_pp)/(n_pos**2-n_pos)
    avg_dist_pn = T.sum(df_pn)/(n_neg**2-n_neg)

    avg_dist_pp = T.clip(avg_dist_pp,epsilon,1.-epsilon)
    avg_dist_pn = T.clip(avg_dist_pn,epsilon,1.-epsilon)

    logloss = -5*T.log(1-avg_dist_pp) - T.log(avg_dist_pn)
    return logloss

def np_f(feat, targets):
    epsilon = 1e-9
    df = np.mean(np.abs(feat[None,:,:] - feat[:,None,:]), axis=2) 
    gt  = targets > 0.5
    non_gt = targets < 0.5
    df_pp = df[gt]
    df_pp = df_pp[:,gt]
    df_pn = df[gt, :]
    df_pn = df_pn[:, non_gt]

    n_pos = np.sum(targets)
    n_neg = np.sum(1-targets)

    avg_dist_pp = np.sum(df_pp)/(n_pos**2-n_pos)
    avg_dist_pn = np.sum(df_pn)/(n_neg**2-n_neg)

    #preds_p = 1-avg_dist_pp
    #preds_n = avg_dist_pn


    #print (5*preds_p+preds_n)/6, 'preds_p', preds_p, 'preds_n', preds_n

    avg_dist_pp = np.clip(avg_dist_pp,epsilon,1.-epsilon)
    avg_dist_pn = np.clip(avg_dist_pn,epsilon,1.-epsilon)

    logloss = -5*np.log(1-avg_dist_pp) - np.log(avg_dist_pn)
    return logloss


tfeat = T.matrix('targets')
ttargets = T.vector()
z = theano_test_fion(tfeat,ttargets)
f = theano.function([tfeat, ttargets], z)


#test 1
feat = np.float32(np.random.rand(4,5))
print feat
targets = np.float32(np.array([0,0,1,1]))
print f(feat, targets )
print np_f(feat, targets)


#test 2
feat = np.ones((4,5), dtype=np.float32)
targets = np.float32(np.array([0,0,1,1]))
print f(feat, targets )
print np_f(feat, targets)


#test 3
feat = np.ones((4,5), dtype=np.float32)
feat[:2] = 0.25 * feat[:2]
feat[2:] = 0.75 * feat[2:]
targets = np.float32(np.array([0,0,1,1]))
#should be very low
print f(feat, targets )
print np_f(feat, targets)

#test 4
feat_noise = 0.1 *np.float32(np.random.rand(4,5))
feat = np.ones((4,5), dtype=np.float32)
feat[:2] = 0.25 * feat[:2]
feat[2:] = 0.75 * feat[2:]
feat = feat + feat_noise
targets = np.float32(np.array([0,0,1,1]))
#should be higer than test 3, but still reasonable
print f(feat, targets )
print np_f(feat, targets)
