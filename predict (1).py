import argparse
import json
import PIL
import torch
import numpy as np
import os

from math import ceil

from  torchvision import models
def gpu():
    if (args.gpu):
        device = 'cuda'
        print("running on gpu")
    else:
        device = 'cpu'
        print("running on cpu")
        
    return 0 

def arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--image_path', type=str, help='path of image to be predicted')
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')
    parser.add_argument('--checkpoint' , type=str, default='flower102_checkpoint.pth', help='path of your saved model')
    parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')
    parser.add_argument('--gpu',action="store_true",help='Use GPU + Cuda for calculations')
    args = parser.parse_args()
    return args

# Function load_checkpoint(checkpoint_path) loads our saved deep learning model from checkpoint
def loadcheckpoint(checkpoint_path):
    """
    Loads deep learning model checkpoint.
    """
    
    if os.path.isfile(checkpoint_path):
       print("=> loading checkpoint '{}'".format(checkpoint_path))
       checkpoint = torch.load(checkpoint_path)
       model = checkpoint['model']
       model.classifier = checkpoint['classifier']
       
       
       #model = model.to(device)
       model.class_to_idx = checkpoint['class_to_idx']
       model.load_state_dict(checkpoint['state_dict'])
       #model = model.(checkpoint['input_size'],checkpoint['output_size'])
       return model
 


def processimage(image_path):
    

    image = PIL.Image.open(image_path)

    # Current dimensions
    width, height = image.size

    # resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    if width < height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
        
    image.thumbnail(size=resize_size)

    # create 224x224 image
    #crop sizes of left,top,right,bottom
    center = width/4, height/4
    left=center[0]-(244/2)
    top=center[1]-(244/2)
    right=center[0]+(244/2)
    bottom =center[1]+(244/2)

    image = image.crop((left, top, right, bottom))

    
    np_image = np.array(image)/255 # imshow() rewuires binary(0,1) so divided by 255

    # Normalize 
    #np_image= transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #this way of normalizing like above doesnt work because Compose object has no transpose attribute which result in
    #bright colored pic    
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
      
   
    # Seting the color to the first channel
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image


def predict(image_tensor, model, device, cat_to_name, topk,image_path):
    image_torch = torch.from_numpy(np.expand_dims(image_tensor,axis=0)).type(torch.FloatTensor)                    
    #image_torch=torch.tensor(processimage(image_path)).float().unsqueeze(0).type(torch.FloatTensor).to("cpu")    
    if (args.gpu):
        image_torch = image_torch.cuda()
        model = model.cuda()    
    
    else:
        image_torch = image_torch.cpu()
        model = model.cpu()

    device = torch.device("cuda" if args.gpu else "cpu")
    model.to(device);
    model.eval();

    # Convert image from numpy to torch
    #image_torch = torch.from_numpy(np.expand_dims(processimage(image_path), 
                                                  #axis=0)).type(torch.FloatTensor).to("cpu")
   
   
    # probabilities-log softmax is on a log scale
    log_pr = model.forward(image_torch)
    #linear scale
    linear_pr = torch.exp(log_pr)
    #Top 5 predictions and labels
    top_pr, top_labels = linear_pr.topk(args.topk)    
    #this could be the bug
    # Detach top predictions into a numpy list
    top_pr = np.array(top_pr.detach())[0]

    top_labels = np.array(top_labels.detach())[0]     

    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[label] for label in top_labels]
    top_fl = [cat_to_name[label] for label in top_labels]
    print(top_pr)
    print(top_fl)

    return top_pr, top_fl,top_labels
            
        
       
   # for i, j in results:
        
        #print("Rank {}:".format(i+1),"Flower: {}, probability: {}%".format(j[1], ceil(j[0]*100)))
  
    


def printprobability(probs, flowers):
    """
    #Converts two lists into a dictionary to print on screen
    """

    for i, j in enumerate(zip(probs, flowers)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, probability: {}%".format(j[0], ceil(j[1]*100)))
           

# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Executing relevant functions
    
    """
    
    global args 
   #Get Keyword Args for Prediction
    args =arg_parser()
    gpu()


    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
        
    # Load model trained with train.py
    model = loadcheckpoint(args.checkpoint)
    
    # Process Image
    image_tensor = processimage(args.image_path)
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_path = args.image_path
    
    prediction = predict(image_tensor, model, device, cat_to_name, args.topk,image_path)
    top_pr, top_fl,top_labels = predict(image_tensor,model, device, cat_to_name, args.topk,image_path)
    
    
    printprobability(top_fl, top_pr)
    
    return prediction



if __name__ == '__main__': main()