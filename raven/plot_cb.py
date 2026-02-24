import torch
import os
import matplotlib.pyplot as plt

def generate_nvsa_codebooks(resume): 
    '''
    Generate the codebooks for NVSA frontend and backend. 
    The codebook can also be loaded if it is stored under args.resume/
    '''
    imfilename = os.path.join(resume,"codebooks.pt")
    if os.path.isfile(imfilename):
        print("Load predefined NVSA codebooks")  
        data= torch.load(imfilename, map_location='cpu')

    perception_cb = data["perception_cb"]
    perception_imdict = data["perception_imdict"]
    backend_cb_cont = data['backend_cb_cont']
    backend_cb_discrete = data['backend_cb_discrete']

    
    return perception_cb, perception_imdict, backend_cb_cont, backend_cb_discrete

def main():
    resumePath = "../Checkpoint_saved/ckpt"
    perception_cb, perception_imdict, backend_cb_cont, backend_cb_discrete = generate_nvsa_codebooks(resumePath)

    plt.plot(perception_cb)
    plt.show()
    plt.savefig("perception_fig.png")

if __name__ == "__main__":
    main()