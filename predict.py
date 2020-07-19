import argparse
import json
import PIL
import torch
import numpy as np

from math import ceil

from  torchvision import models


def arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--image_path', type=str, help='path of image to be predicted')
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')
    parser.add_argument('--checkpoint' , type=str, default='flower102_checkpoint.pth', help='path of your saved model')
    parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')
    parser.add_argument('--gpu',action="store_true",help='Use GPU + Cuda for calculations')
    
    
    
    #parser = argparse.ArgumentParser(description="Neural Network ")
    #parser.add_argument('image', type=str, help='log image path.')
                      
    # Load checkpoint 

    #parser.add_argument('checkpoint', type=str,help='log checkpoint path.')
    #parser.add_argument("-c", "--checkpoint", help="checkpoint filename to restore", required=True, dest="checkpoint")
    #parser.add_argument('--top_k',type=int,help='Choose top K matches as int.')
    #parser.add_argument("-p", "--print-top-k", help="Print top-K classes", default="5", dest="print_top_k")
    #parser.add_argument('--category_names',type=str,help='Mapping from categories to real names.',)
    #parser.add_argument('cat_to_name.json')
    #parser.add_argument('--gpu',action="store_true",help='Use GPU + Cuda for calculations')
    
    
    #parser.add_argument("-i", "--image", help="image filename to read.", required=True, dest="image")
    #parser.add_argument("-c", "--checkpoint", help="checkpoint filename to restore", required=True, dest="checkpoint")
    
    #parser.add_argument("-g", "--gpu", help="GPU training if available, default gpu = True", default=True, dest="gpu")
    #parser.add_argument("-j", "--cat-json", help="json filename with categories to map classnames", required=True, dest="cat_json")
    args = parser.parse_args()
    return args

# Function load_checkpoint(checkpoint_path) loads our saved deep learning model from checkpoint
def loadcheckpoint(checkpoint_path):
    """
    Loads deep learning model checkpoint.
    """
        # Load the saved file
    checkpoint = torch.load(checkpoint_path)
    model = models.vgg16(pretrained=True)
    
 
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    # Load stuff from checkpoint
    #model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint.get('state_dict'))
    
    return model    
    # Download pretrained model
    #model = models.vgg16(pretrained=True);
    #model.to(device)
    # Load the saved file
    #checkpoint = torch.load(checkpoint_path)
    #model.arch = checkpoint.get('arch')
    #assert checkpoint.get("arch") == model.__module__.split('.')[-1]

    
    # Freeze parameters so we don't backprop through them
    #for param in model.parameters(): param.requires_grad = False
    
    # Load stuff from checkpoint
    #model = models.densenet121(pretrained=true)
    #checkpoint = torch.load(check.pth)
    #model.classifier = checkpoint['classifier']
    #model.load_state_dict(checkpoint['state_dict'])
    #model.load_state_dict(checkpoint.get('state_dict'))
    
    #return model


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
    model.to("cpu")
    model.eval();

    # Convert image from numpy to torch
    #image_torch = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  #axis=0)).type(torch.FloatTensor).to("cpu")
    image_torch=torch.tensor(processimage(image_path)).float().unsqueeze(0).type(torch.FloatTensor).to("cpu")
        #image_torch=torch.tensor(process_image(image_path)).float().unsqueeze(0).type(torch.FloatTensor).to("cpu")
    # probabilities-log softmax is on a log scale
    #log_pr = model.forward(image_torch)
    #linear scale
    #linear_pr = torch.exp(log_pr)
    #Top 5 predictions and labels
    #top_pr, top_labels = linear_pr.topk(top_k)    
    #this could be the bug
    # Detach top predictions into a numpy list
    #top_pr = np.array(top_pr.detach())[0]

    #top_labels = np.array(top_labels.detach())[0]     

    #idx_to_class = {val: key for key, val in    
                                      #model.class_to_idx.items()}
    #top_labels = [idx_to_class[label] for label in top_labels]
    #top_fl = [cat_to_name[label] for label in top_labels]
    
    #return top_pr, top_labels, top_fl    
    #if (args.gpu):
        #image = image.cuda()
        #model = model.cuda()    
    
    #else:
        #image = image.cpu()
        #model = model.cpu()

    
    topk=int(topk)
    outputs = model(image_torch)
    probs, classes = torch.exp(outputs).topk(topk)
    probs, classes = probs[0].tolist(), classes[0].add(1).tolist()
    results = zip(probs,classes)
    return results
def read_c():

    if (args.cat_to_name is not None):
        cat_file = args.cat_to_name 
        jfile = json.loads(open(cat_file).read())
        return jfile
    return None    


def printprobability(results):#probs, flowers):
    """
    Converts two lists into a dictionary to print on screen
    """

    cat_file =read_c()
    i = 0
    for prob, cla in results:
        i = i + 1
        prob = str(round(prob,4) * 100.) + '%'
        if (cat_file):
            cla = cat_file.get(str(cla),'None')#str(c),'None')
        else:
            cla = ' class {}'.format(str(cla))
        print("{}.{} ({})".format(i, cla,prob))
    return None
    #for i, j in enumerate(zip(flowers, probs)):
        #print ("Rank {}:".format(i+1),
               #"Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))
    
           

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
    
    #parser = argparse.ArgumentParser()
    #parser.add_argument('cat_to_name.json')
    #args = parser.parse_args()
    

    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
        
    # Load model trained with train.py
    model = loadcheckpoint(args.checkpoint)
    
    # Process Image
    image_tensor = processimage(args.image_path)
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_path = args.image_path
    prediction = predict(image_tensor, model, device, cat_to_name, args.topk,image_path)
    printprobability(prediction)
    return prediction

    # Use `processed_image` to predict the top K most likely classes
    #top_probs, top_labels, top_flowers = predict(image_tensor, model, 
                                                 #device, cat_to_name,
                                                # args.topk,args.image_path)
    
    # Print out probabilities
    #print_probability(top_flowers, top_probs)

if __name__ == '__main__': main()