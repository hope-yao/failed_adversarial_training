import numpy as np

def track_absolute_diff(sess, model, x_batch, x_batch_adv, y_batch):
    model_fix = model
    x_batch_nat = x_batch
    batch_size = x_batch.shape[0]

    nat_dict = {model_fix.x_input: x_batch_nat,
                model_fix.y_input: y_batch}
    adv_dict = {model_fix.x_input: x_batch_adv,
                model_fix.y_input: y_batch}

    diff_input = np.mean(
        np.abs(sess.run(model_fix.x_input, feed_dict=adv_dict) - sess.run(model_fix.x_input, feed_dict=nat_dict)))
    diff_conv1 = np.mean(
        np.abs(sess.run(model_fix.h_conv1, feed_dict=adv_dict) - sess.run(model_fix.h_conv1, feed_dict=nat_dict)))
    diff_conv2 = np.mean(
        np.abs(sess.run(model_fix.h_conv2, feed_dict=adv_dict) - sess.run(model_fix.h_conv2, feed_dict=nat_dict)))
    diff_fc1 = np.mean(
        np.abs(sess.run(model_fix.h_fc1, feed_dict=adv_dict) - sess.run(model_fix.h_fc1, feed_dict=nat_dict)))
    diff_fc2 = np.mean(np.abs(
        sess.run(model_fix.pre_softmax, feed_dict=adv_dict) - sess.run(model_fix.pre_softmax, feed_dict=nat_dict)))

    l2_diff = [diff_input, diff_conv1, diff_conv2, diff_fc1, diff_fc2]
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # plt.plot([diff_input, diff_conv1, diff_conv2, diff_fc1, diff_fc2], 'b.')
    # # plt.axis([-0.1,4.1,0,3])

    nat_dict = {model_fix.x_input: x_batch_nat,
                model_fix.y_input: y_batch}
    adv_dict = {model_fix.x_input: x_batch_adv,
                model_fix.y_input: y_batch}

    fea_nat = sess.run(model_fix.x_input, feed_dict=nat_dict)
    fea_adv = sess.run(model_fix.x_input, feed_dict=adv_dict)
    err = []
    for i in range(batch_size):
        err = np.max(np.abs(fea_nat[i].flatten() - fea_adv[i].flatten()))
    diff_x_input = np.mean(err)

    fea_nat = sess.run(model_fix.h_conv1, feed_dict=nat_dict)
    fea_adv = sess.run(model_fix.h_conv1, feed_dict=adv_dict)
    err = []
    for i in range(batch_size):
        err = np.max(np.abs(fea_nat[i].flatten() - fea_adv[i].flatten()))
    diff_h_conv1 = np.mean(err)

    fea_nat = sess.run(model_fix.h_conv2, feed_dict=nat_dict)
    fea_adv = sess.run(model_fix.h_conv2, feed_dict=adv_dict)
    err = []
    for i in range(batch_size):
        err = np.max(np.abs(fea_nat[i].flatten() - fea_adv[i].flatten()))
    diff_h_conv2 = np.mean(err)

    fea_nat = sess.run(model_fix.h_fc1, feed_dict=nat_dict)
    fea_adv = sess.run(model_fix.h_fc1, feed_dict=adv_dict)
    err = []
    for i in range(batch_size):
        err = np.max(np.abs(fea_nat[i].flatten() - fea_adv[i].flatten()))
    diff_fc1 = np.mean(err)

    fea_nat = sess.run(model_fix.pre_softmax, feed_dict=nat_dict)
    fea_adv = sess.run(model_fix.pre_softmax, feed_dict=adv_dict)
    err = []
    for i in range(batch_size):
        err = np.max(np.abs(fea_nat[i].flatten() - fea_adv[i].flatten()))
    diff_pre_softmax = np.mean(err)

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # plt.plot([diff_x_input, diff_h_conv1, diff_h_conv2, diff_fc1, diff_pre_softmax], 'r.')
    # # plt.axis([-0.1,4.1,0,3])
    linf_diff = [diff_x_input, diff_h_conv1, diff_h_conv2, diff_fc1, diff_pre_softmax]

    return l2_diff, linf_diff

def track_relative_diff(sess, model, x_batch, x_batch_adv, y_batch):
    model_fix = model
    x_batch_nat = x_batch
    batch_size = x_batch.shape[0]

    nat_dict = {model_fix.x_input: x_batch_nat,
                model_fix.y_input: y_batch}
    adv_dict = {model_fix.x_input: x_batch_adv,
                model_fix.y_input: y_batch}

    diff_input = np.mean(np.abs(
        sess.run(model_fix.x_input, feed_dict=adv_dict) - sess.run(model_fix.x_input, feed_dict=nat_dict))) / np.mean(
        np.abs(sess.run(model_fix.x_input, feed_dict=nat_dict)))
    diff_conv1 = np.mean(np.abs(
        sess.run(model_fix.h_conv1, feed_dict=adv_dict) - sess.run(model_fix.h_conv1, feed_dict=nat_dict))) / np.mean(
        np.abs(sess.run(model_fix.h_conv1, feed_dict=nat_dict)))
    diff_conv2 = np.mean(np.abs(
        sess.run(model_fix.h_conv2, feed_dict=adv_dict) - sess.run(model_fix.h_conv2, feed_dict=nat_dict))) / np.mean(
        np.abs(sess.run(model_fix.h_conv2, feed_dict=nat_dict)))
    diff_fc1 = np.mean(np.abs(
        sess.run(model_fix.h_fc1, feed_dict=adv_dict) - sess.run(model_fix.h_fc1, feed_dict=nat_dict))) / np.mean(
        np.abs(sess.run(model_fix.h_fc1, feed_dict=nat_dict)))
    diff_fc2 = np.mean(np.abs(sess.run(model_fix.pre_softmax, feed_dict=adv_dict) - sess.run(model_fix.pre_softmax,
                                                                                             feed_dict=nat_dict))) / np.mean(
        np.abs(sess.run(model_fix.pre_softmax, feed_dict=nat_dict)))

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # plt.plot([diff_input, diff_conv1, diff_conv2, diff_fc1, diff_fc2], 'b.', label='L_1')
    # # plt.axis([-0.1,4.1,0,3])
    l2_diff = [diff_input, diff_conv1, diff_conv2, diff_fc1, diff_fc2]

    nat_dict = {model_fix.x_input: x_batch_nat,
                model_fix.y_input: y_batch}
    adv_dict = {model_fix.x_input: x_batch_adv,
                model_fix.y_input: y_batch}

    fea_nat = sess.run(model_fix.x_input, feed_dict=nat_dict)
    fea_adv = sess.run(model_fix.x_input, feed_dict=adv_dict)
    err = []
    for i in range(batch_size):
        err = np.max(np.abs(fea_nat[i].flatten() - fea_adv[i].flatten())) / np.max(fea_nat[i].flatten())
    diff_x_input = np.mean(err)

    fea_nat = sess.run(model_fix.h_conv1, feed_dict=nat_dict)
    fea_adv = sess.run(model_fix.h_conv1, feed_dict=adv_dict)
    err = []
    for i in range(batch_size):
        err = np.max(np.abs(fea_nat[i].flatten() - fea_adv[i].flatten())) / np.max(fea_nat[i].flatten())
    diff_h_conv1 = np.mean(err)

    fea_nat = sess.run(model_fix.h_conv2, feed_dict=nat_dict)
    fea_adv = sess.run(model_fix.h_conv2, feed_dict=adv_dict)
    err = []
    for i in range(batch_size):
        err = np.max(np.abs(fea_nat[i].flatten() - fea_adv[i].flatten())) / np.max(fea_nat[i].flatten())
    diff_h_conv2 = np.mean(err)

    fea_nat = sess.run(model_fix.h_fc1, feed_dict=nat_dict)
    fea_adv = sess.run(model_fix.h_fc1, feed_dict=adv_dict)
    err = []
    for i in range(batch_size):
        err = np.max(np.abs(fea_nat[i].flatten() - fea_adv[i].flatten())) / np.max(fea_nat[i].flatten())
    diff_fc1 = np.mean(err)

    fea_nat = sess.run(model_fix.pre_softmax, feed_dict=nat_dict)
    fea_adv = sess.run(model_fix.pre_softmax, feed_dict=adv_dict)
    err = []
    for i in range(batch_size):
        err = np.max(np.abs(fea_nat[i].flatten() - fea_adv[i].flatten())) / np.max(fea_nat[i].flatten())
    diff_pre_softmax = np.mean(err)

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # plt.plot([diff_x_input, diff_h_conv1, diff_h_conv2, diff_fc1, diff_pre_softmax], 'r.', label='L_inf')
    # plt.legend()
    linf_diff = [diff_x_input, diff_h_conv1, diff_h_conv2, diff_fc1, diff_pre_softmax]
    return l2_diff, linf_diff
