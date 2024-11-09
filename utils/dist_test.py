import torch.distributed as dist


def print_one():
    print("return 1")
    dist.destroy_process_group()
    return 1

def test():
    group = dist.init_process_group("gloo")
    dist.barrier()
    print(f"hello from {dist.get_rank()}")

    if dist.get_rank()==0:
        results =print_one()
        print(results)


if __name__=="__main__":
    test()
