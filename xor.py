def lcg(modulus, a, c, seed):
    """Linear congruential generator."""
    while True:
        seed = (a * seed + c) % modulus
        yield seed

#random num gen
modulus = 2**32
a = 1103515245
c = 12345
seed = 123456789
rand = lcg(modulus, a, c, seed)

def pseudo_random():
    return next(rand) / modulus * 2 - 1

def sigmoid(x):
    return 2 / (1 + 2.71828 ** (-2 * x)) - 1

def apply_sigmoid(matrix):
    return [[sigmoid(element) for element in row] for row in matrix]

def sigmoid_output_to_derivative(output):
    return output * (1 - output)

def apply_sigmoid_derivative(matrix):
    return [[sigmoid_output_to_derivative(element) for element in row] for row in matrix]

# Matrix operations
def matmul(a, b):
    result = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result

def matadd(a, b):
    result = [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    return result

def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def subtract(a, b):
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def scalar_multiply(matrix, scalar):
    if isinstance(scalar, list):
        scalar = scalar[0][0]
    return [[element * scalar for element in row] for row in matrix]

inputs = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
expected_output = [[[0]], [[1]], [[1]], [[0]]]
print("Expected Results: ", expected_output)

epochs = 10000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1
hidden_weights = [[pseudo_random() for _ in range(hiddenLayerNeurons)] for _ in range(inputLayerNeurons)]
hidden_bias = [[pseudo_random() for _ in range(hiddenLayerNeurons)]]
output_weights = [[pseudo_random() for _ in range(outputLayerNeurons)] for _ in range(hiddenLayerNeurons)]
output_bias = [[pseudo_random() for _ in range(outputLayerNeurons)]]

#train
for _ in range(epochs):
    for x, y in zip(inputs, expected_output):
        #forward prop
        hidden_layer_input = matadd(matmul(x, hidden_weights), hidden_bias)
        hidden_layer_output = apply_sigmoid(hidden_layer_input)
        output_layer_input = matadd(matmul(hidden_layer_output, output_weights), output_bias)
        predicted_output = apply_sigmoid(output_layer_input)

        #back prop
        error = subtract(y, predicted_output)
        d_predicted_output = scalar_multiply(apply_sigmoid_derivative(predicted_output), error)
        error_hidden_layer = matmul(d_predicted_output, transpose(output_weights))
        d_hidden_layer = scalar_multiply(apply_sigmoid_derivative(hidden_layer_output), error_hidden_layer)

        output_weights = matadd(output_weights, scalar_multiply(matmul(transpose(hidden_layer_output), d_predicted_output), lr))
        output_bias = matadd(output_bias, scalar_multiply(d_predicted_output, lr))
        hidden_weights = matadd(hidden_weights, scalar_multiply(matmul(transpose(x), d_hidden_layer), lr))
        hidden_bias = matadd(hidden_bias, scalar_multiply(d_hidden_layer, lr))

#test
test_results = [apply_sigmoid(matadd(matmul(apply_sigmoid(matadd(matmul(input, hidden_weights), hidden_bias)), output_weights), output_bias)).pop().pop() for input in inputs]
test_results_rounded = [round(result) for result in test_results]
print("Test Results:", test_results_rounded)