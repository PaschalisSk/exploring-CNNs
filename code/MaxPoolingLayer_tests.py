import numpy as np
from mlp.layers import MaxPooling2DLayer
test_correct_target = np.load("../data/max_pooling_correct.npz")
test_inputs = test_correct_target['test_inputs']
test_grads_wrt_outputs = test_correct_target['test_grads_wrt_outputs']
layer_to_test = MaxPooling2DLayer(input_height=test_inputs.shape[2], input_width=test_inputs.shape[3], size=2, stride=2)
fprop_preds = layer_to_test.fprop(inputs=test_inputs)

fprop_accuracy = np.mean(np.allclose(test_correct_target['fprop_correct'], fprop_preds))

if fprop_accuracy == 1.0:
    print("Passed fprop test")
else:
    print("Failed fprop test, correct outputs: {}, predicted_outputs: {}".format(test_correct_target['fprop_correct'], fprop_preds))

bprop_preds = layer_to_test.bprop(inputs=test_inputs, outputs=fprop_preds, grads_wrt_outputs=test_grads_wrt_outputs)
bprop_accuracy = np.mean(np.allclose(test_correct_target['bprop_correct'], bprop_preds))

if fprop_accuracy == 1.0:
    print("Passed fprop test")
else:
    print("Failed fprop test, correct outputs: {}, predicted_outputs: {}".format(test_correct_target['fprop_correct'], fprop_preds))

if bprop_accuracy == 1.0:
    print("Passed bprop test")
else:
    print("Failed bprop test, correct grads: {}, predicted grads: {}".format(test_correct_target['bprop_correct'], bprop_preds))
