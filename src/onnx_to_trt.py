import tensorrt as trt
import os

ONNX_PATH   = "scrfd_2.5g_bnkps.onnx"
ENGINE_PATH = "scrfd_2.5g_fp16.trt"
INPUT_SHAPE = (1, 3, 640, 640)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build():
    print(f"TensorRT version: {trt.__version__}")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    parser  = trt.OnnxParser(network, TRT_LOGGER)

    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return

    print("ONNX parsed successfully")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # Try different ways to enable FP16
    try:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled (method 1)")
    except:
        try:
            config.flags = 1 << 0   # FP16 flag value
            print("FP16 enabled (method 2)")
        except:
            print("Could not enable FP16, using FP32")

    # Optimization profile
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    print(f"Input name: {input_name}")
    profile.set_shape(input_name, INPUT_SHAPE, INPUT_SHAPE, INPUT_SHAPE)
    config.add_optimization_profile(profile)

    print("Building engine... please wait")
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        print("Build failed")
        return

    with open(ENGINE_PATH, "wb") as f:
        f.write(engine)

    print(f"\nSuccess! Saved → {ENGINE_PATH}")
    print(f"Size: {os.path.getsize(ENGINE_PATH)/1024/1024:.2f} MB")

if __name__ == "__main__":
    build()