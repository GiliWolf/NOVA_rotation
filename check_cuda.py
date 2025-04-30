import torch

def check_cuda():
    if torch.cuda.is_available():
        print(f"CUDA is available! {torch.cuda.device_count()} GPU(s) detected.")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    check_cuda()
