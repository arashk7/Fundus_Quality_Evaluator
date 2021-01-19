import os
import pytorch_lightning as pl
from PIL import Image
import torch
import torchvision.transforms as transforms
from fqe_eff.fqe_isbi import FQEModel,Dataset_ISBI

'''params'''
img_size = 300
batch_size=4

ISBI_TRAIN_PATH = 'E:\Dataset\DR\DeepDr\merged_tr_vl'
ISBI_TEST_PATH = 'E:\Dataset\DR\DeepDr\Onsite-Challenge1-2-Evaluation'


'''Sample image path'''
image_path = "E:\Dataset\DR\DeepDr\Onsite-Challenge1-2-Evaluation/16/16_l1.jpg"

''' Instanciate an object from FQEModel'''
model = FQEModel()

''' Loading weights'''
model.eff_model.load("checkpoints_isbi/_ckpt_epoch_2.ckpt")
# Load state_dict
# model.load_state_dict(torch.load('checkpoints_isbi/_ckpt_epoch_2.ckpt'))
# model.load_from_checkpoint("checkpoints_isbi/_ckpt_epoch_2.ckpt")#'checkpoints_isbi/fqe_isbi_acc_1.0_fold_2.pt')
model.freeze()

def test_isbi(trainer, model, dataset_test):
    test_loader = torch.utils.data.DataLoader(dataset_test,  # num_workers= 16,
                                              batch_size=batch_size)
    result = trainer.test(model, test_loader)
    acc = result[0]['test_epoch_acc']
    acc = round(acc, 3)
    print('>>>>>>>>>>>>>>>>Test acc: ' + str(acc))


'''Transform the image'''
transform_test = transforms.Compose([transforms.Resize((img_size, img_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]),

                                     ])
isbi_dataset_test = Dataset_ISBI(os.path.join(ISBI_TEST_PATH, 'Onsite-Challenge1-2-Evaluation_full.csv'),
                                     ISBI_TEST_PATH,
                                     transform=transform_test)
trainer = pl.Trainer(gpus=1)

test_isbi(trainer, model, isbi_dataset_test)


'''Load a test image'''
img = Image.open(image_path)



img = transform_test(img)
# img2 = transform_test(img)
# img3 = transform_test(img)
# img4 = transform_test(img)


''' Unsqueeze batch dimension, in case you are dealing with a single image'''
img = img.unsqueeze(0)
# batch = torch.Tensor([img,img,img,img])
# batch = next(iter([img,img,img,img]))
c = torch.cat((img, img,img,img), 0)


''' Predict the quality of Fundus image'''
out = model.predict(c)
pred = torch.reshape(out.argmax(dim=1, keepdim=True), ( 1,4))

print(pred)


