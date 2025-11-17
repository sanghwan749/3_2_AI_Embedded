import numpy as np
import time
import tflite_runtime.interpreter as tflite
from sklearn.metrics import precision_score, recall_score

# -----------------------------------------------------------
# MNIST LOAD FUNCTIONS (same as in tflite_tpu_sample.py)
# -----------------------------------------------------------
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

# -----------------------------------------------------------
# Preprocess helper
# -----------------------------------------------------------
def preprocess_image(img, input_details):
    img = img.astype(np.float32) / 255.0

    # reshape according to model requirement
    expected_shape = input_details[0]['shape']
    dtype = input_details[0]['dtype']
    quant = input_details[0].get('quantization', (0.0, 0))
    scale, zero_point = quant if quant else (0.0, 0)

    x = img.reshape(1, 28, 28, 1)

    # Quantization
    if np.issubdtype(dtype, np.integer):
        if scale == 0:
            x = np.clip(x * 255.0, 0, 255).astype(dtype)
        else:
            x = np.round(x / scale + zero_point).astype(dtype)
    else:
        x = x.astype(dtype)

    return np.ascontiguousarray(x)

# -----------------------------------------------------------
# Inference loop
# -----------------------------------------------------------
def run_inference(interpreter, images, labels, num_iter=200):
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    latencies = []
    preds = []
    trues = []

    for i in range(num_iter):
        img = images[i]
        label = labels[i]

        x_input = preprocess_image(img, input_details)

        interpreter.set_tensor(input_index, x_input)

        start = time.time()
        interpreter.invoke()
        end = time.time()

        latencies.append((end - start) * 1000)  # ms

        # get output
        out_raw = interpreter.get_tensor(output_index)[0]
        scale, zero_point = output_details[0].get('quantization', (0.0, 0))

        if np.issubdtype(out_raw.dtype, np.integer) and scale != 0:
            out = (out_raw.astype(np.float32) - zero_point) * scale
        else:
            out = out_raw

        pred = int(np.argmax(out))
        preds.append(pred)
        trues.append(label)

    preds = np.array(preds)
    trues = np.array(trues)

    precision = precision_score(trues, preds, average='micro')  
    recall = recall_score(trues, preds, average='micro')

    return precision, recall, np.mean(latencies)

# -----------------------------------------------------------
# Main Benchmark
# -----------------------------------------------------------
def main():
    # Load MNIST
    images = load_mnist_images("t10k-images-idx3-ubyte")
    labels = load_mnist_labels("t10k-labels-idx1-ubyte")

    print("Loaded MNIST test set.")

    # -------------------------------
    # CPU INTERPRETER
    # -------------------------------
    cpu_model = "mnist_lenet_quant.tflite"
    cpu_interpreter = tflite.Interpreter(model_path=cpu_model)

    print("\nRunning CPU inference 200 times...")
    cpu_precision, cpu_recall, cpu_latency = run_inference(cpu_interpreter, images, labels)

    print("\n=== CPU RESULTS (200 runs) ===")
    print(f"Average Precision: {cpu_precision:.4f}")
    print(f"Average Recall:    {cpu_recall:.4f}")
    print(f"Average Latency:   {cpu_latency:.3f} ms")

    # -------------------------------
    # EDGE TPU INTERPRETER
    # -------------------------------
    tpu_model = "mnist_lenet_quant_edgetpu.tflite"

    try:
        delegate = tflite.load_delegate("libedgetpu.so.1")
    except Exception as e:
        print("\n[ERROR] Cannot load EdgeTPU delegate:", e)
        return

    tpu_interpreter = tflite.Interpreter(
        model_path=tpu_model,
        experimental_delegates=[delegate]
    )

    print("\nRunning EdgeTPU inference 200 times...")
    tpu_precision, tpu_recall, tpu_latency = run_inference(tpu_interpreter, images, labels)

    print("\n=== EDGE TPU RESULTS (200 runs) ===")
    print(f"Average Precision: {tpu_precision:.4f}")
    print(f"Average Recall:    {tpu_recall:.4f}")
    print(f"Average Latency:   {tpu_latency:.3f} ms")

    print("\n=== SUMMARY ===")
    print(f"CPU  Latency: {cpu_latency:.3f} ms")
    print(f"TPU  Latency: {tpu_latency:.3f} ms")
    print(f"Speedup (CPU / TPU): {cpu_latency / tpu_latency:.2f}x")

# -----------------------------------------------------------
if __name__ == "__main__":
    main()
