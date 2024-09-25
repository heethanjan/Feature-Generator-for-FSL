import time
import clip
import numpy as np
import torch
import numpy as np
import torch
import clip
import pickle
import numpy as np
import os
from scipy.spatial import distance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle

dict_train =  {
    '52': [
        'The unicycle is a single-wheeled vehicle with a seat and pedals.',
        'It has a circular shape with a tall central wheel.',
        'The unicycle features a handlebar for balance and maneuverability.'
    ],
    '59': [
        'The hotdog is a long, cylindrical-shaped food item.',
        'It typically consists of a sausage wrapped in a soft bun.',
        'It is often topped with condiments such as ketchup, mustard, and onions.'
    ],
    '41': [
        'The photocopier is a machine used for duplicating documents.',
        'It has a rectangular shape with a flat glass surface.',
        'The photocopier features buttons and controls for adjusting settings.'
    ],
    '16': [
        'The miniature poodle is a small-sized breed of dog.',
        'It has a fluffy and curly coat that comes in various colors.',
        'The miniature poodle has a distinct round face and expressive eyes.'
    ],
    '8': [
        'The Walker hound is a breed of dog known for its hunting abilities.',
        'It has a slim and athletic body with short fur.',
        'The Walker hound features long ears and a strong sense of smell.'
    ],
    '13': [
        'The Tibetan Mastiff is a large and powerful breed of dog.',
        'It has a muscular build and a thick double coat.',
        'The Tibetan Mastiff has a broad head and a bushy tail.'
    ],
    '39': [
        'The parallel bars are gymnastics equipment used for balancing and strength exercises.',
        'They consist of two horizontal bars set parallel to each other at different heights.',
        'The parallel bars provide support for various acrobatic movements.'
    ],
    '50': [
        'The tile roof is a type of roofing made of clay or concrete tiles.',
        'It has a sloping shape, and the tiles overlap to create a waterproof barrier.',
        'The tile roof is durable and provides insulation against heat and cold.'
    ],
    '7': [
        'The dugong, also known as the "sea cow," is a marine mammal.',
        'It has a large, cylindrical body and a rounded tail.',
        'The dugong features paddle-like flippers and a snout with bristly whiskers.'
    ],
    '26': [
        'The chime is a musical instrument consisting of suspended metal tubes or rods.',
        'It has a vertical, elongated shape with different lengths of tubes.',
        'When struck, the chime produces resonant and harmonious tones.'
    ],
    '24': [
        'The beer bottle is a container used for storing and serving beer.',
        'It has a cylindrical shape with a narrow neck and a cap or a crown seal.',
        'The beer bottle may feature labels or embossed designs for branding.'
    ],
    '31': [
        'The drawer is a storage compartment typically found in furniture like desks and dressers.',
        'It has a rectangular shape with a handle or knob for pulling.',
        'The drawer slides in and out of a larger structure for easy access to stored items.'
    ],
    '21': [
        'The aircraft carrier is a large warship used for deploying and recovering aircraft.',
        'It has a flat, elongated deck with a superstructure and multiple flight decks.',
        'The aircraft carrier features a vast hangar for storing and maintaining aircraft.'
    ],
       '12': [
        'The Boxer is a medium-sized breed of dog with a muscular and athletic build.',
        'It has a square-shaped head and a strong jawline.',
        'The Boxer is known for its playful and energetic nature.'
    ],
    '63': [
        'Corn, also known as maize, is a tall and slender cereal plant.',
        'It has a cylindrical shape with rows of edible kernels tightly packed on a cob.',
        'Corn plants feature long leaves and produce tassels that contain pollen.'
    ],
    '2': [
        'The Triceratops is a large dinosaur that lived during the Late Cretaceous period.',
        'It has a robust body with a massive head adorned with three horns and a bony frill.',
        'The Triceratops features a beak-like mouth and a plated skin texture.'
    ],
    '18': [
        'The ladybug, also known as a ladybird, is a small and colorful beetle.',
        'It has a round or oval shape with a dome-like back and short legs.',
        'The ladybug features distinctive patterns and bright colors on its wing covers.'
    ],
    '0': [
        'The Common Rosefinch is a small bird with a compact and plump body.',
        'It has a short beak and a rounded head.',
        'The Common Rosefinch is known for its vibrant plumage, especially in males.'
    ],
    '14': [
        'The French Bulldog is a small-sized breed of dog with a sturdy and muscular build.',
        'It has a compact body with a distinctive pushed-in nose and bat-like ears.',
        'The French Bulldog features a short and smooth coat that comes in various colors.'
    ],
    '6': [
        'The jellyfish is a gelatinous marine creature with a bell-shaped or umbrella-shaped body.',
        'It has long, trailing tentacles that contain venomous stinging cells.',
        'Jellyfish come in different colors and sizes, and they move by pulsating their bodies.'
    ],
    '30': [
        'The dome is a hemispherical or half-spherical architectural structure.',
        'It has a curved shape that forms a roof or ceiling.',
        'Domes are often used in buildings, observatories, and religious structures.'
    ],
    '27': [
        'The clog is a type of footwear with a thick, sturdy sole.',
        'It has a wooden or rubber base and an open back or straps.',
        'Clogs are known for their practicality, durability, and often feature decorative elements.'
    ],
    '19': [
        'The three-toed sloth is a slow-moving mammal found in the rainforests of Central and South America.',
        'It has a small body with long limbs and three toes on its forelimbs.',
        'The three-toed sloth has shaggy fur and hangs upside down from tree branches.'
    ],
    ' 5': [
        'The Toco Toucan is a large bird species known for its vibrant and oversized beak.',
        'It has a black body with white and yellow plumage on its face and chest.',
        'The Toco Toucan features a distinctive call and is native to South America.'
    ],
    '22': [
        'The garbage bin, also known as a trash can or waste bin, is a container for collecting and storing garbage.',
        'It has a cylindrical or rectangular shape with a lid for keeping the contents enclosed.',
        'Garbage bins are made from durable materials and often have handles for easy transportation.'
    ],
    '3': [
        'The green mamba is a highly venomous snake known for its striking green coloration.',
        'It has a long and slender body with smooth scales.',
        'The green mamba is agile and fast, with excellent tree-climbing abilities.'
    ],
    '62': [
        'The bolete is a type of edible mushroom found in forests and woodlands.',
        'It has a rounded cap with a spongy texture underneath and a thick stem.',
        'Boletes come in various colors, and some species have a distinct network pattern on their caps.'
    ],
    '29': [
        'The prayer rug is a small carpet used by Muslims for praying.',
        'It has a rectangular shape and is often adorned with intricate patterns and designs.',
        'Prayer rugs provide a clean and comfortable surface for prayer and are easily portable.'
    ],
    '11': [
        'The Komondor is a large and powerful breed of dog with a distinctive corded coat.',
        'It has a robust body and a large head with a dense mass of cords that resemble dreadlocks.',
        'The Komondor is known for its protective nature and strong herding instincts.'
    ],
    '10': [
        'The Gordon Setter is a medium to large-sized breed of dog with a strong and athletic build.',
        'It has a long, silky coat that is mostly black with tan markings on the chest, legs, and face.',
        'The Gordon Setter is known for its keen sense of smell and its ability to excel in various dog sports.'
    ],
    '61': [
        'A cliff is a steep rock face or slope found along coastlines, mountains, or canyons.',
        'It has a vertical or near-vertical shape, often formed by erosion or tectonic activity.',
        'Cliffs offer breathtaking views and can be challenging to climb or navigate.'
    ],
    '45': [
        'A snorkel is a flexible tube used by swimmers and divers to breathe while their faces are submerged in water.',
        'It has a tube-like shape with a mouthpiece for comfortable breathing.',
        'Snorkels allow users to observe underwater life and explore shallow waters.'
    ],
    '49': [
        'A war tank, also known as a battle tank or armored tank, is a heavily armored military vehicle.',
        'It has a large and robust body with caterpillar tracks for mobility.',
        'War tanks are equipped with powerful weapons and provide protection for the crew.'
    ],
    '32': [
        'A fire screen, also called a fireplace screen or spark guard, is a protective barrier placed in front of a fireplace.',
        'It has a flat or folding shape and is made of metal or glass.',
        'Fire screens prevent sparks and embers from escaping the fireplace and can add decorative elements to the hearth.'
    ],
        '43': [
        'A reel is a cylindrical device used for winding and storing various materials, such as thread, wire, or film.',
        'It has a circular shape with flanges on the sides to prevent the material from slipping off.',
        'Reels can be made of different materials, including plastic, metal, or wood.'
    ],
    '1': [
        'The American Robin is a medium-sized songbird with a plump body and a reddish-orange breast.',
        'It has a rounded head and a long, thin beak for foraging insects and fruits.',
        'The American Robin is known for its melodious song and is a common sight in North America.'
    ],
    '58': [
        'Consommé is a clear and flavorful soup made by clarifying and straining a rich stock.',
        'It has a liquid consistency and is often served hot as a starter or base for other dishes.',
        'Consommé is known for its delicate taste and is typically garnished with herbs or vegetables.'
    ],
    '28': [
        'A cocktail shaker is a container used for mixing and chilling beverages, especially cocktails.',
        'It has a cylindrical or conical shape with a tight-fitting lid and a built-in strainer.',
        'Cocktail shakers are commonly made of metal and are essential tools for bartenders.'
    ],
    '53': [
        'An upright piano is a musical instrument with a vertical string arrangement.',
        'It has a rectangular or slightly curved shape and stands on its own legs.',
        'Upright pianos produce rich and resonant sounds and are often used in homes, schools, and music studios.'
    ],
    '42': [
        'A carpet is a textile floor covering that is typically made of woven or tufted fibers.',
        'It has a flat and soft surface, providing comfort and insulation.',
        'Carpets come in various patterns, colors, and textures and can be used to enhance the aesthetics of a room.'
    ],
    '46': [
        'A solar dish is a device used to concentrate sunlight onto a receiver to generate heat or electricity.',
        'It has a concave shape that focuses sunlight onto a specific point.',
        'Solar dishes are often used in solar power systems and can harness renewable energy.'
    ],
    '57': [
        'Street signs are traffic signs placed along roads and streets to provide information and regulate traffic.',
        'They come in various shapes, sizes, and colors and display symbols, text, or a combination of both.',
        'Street signs include stop signs, speed limit signs, directional signs, and many others.'
    ],
    '25': [
        'A carousel, also known as a merry-go-round, is an amusement ride with a rotating platform.',
        'It has a circular shape with colorful seats or animals that move up and down as the carousel spins.',
        'Carousels are popular attractions in parks and fairs, especially among children.'
    ],
    '23': [
        'Barrels are cylindrical containers with a curved top and bottom, usually made of wood or metal.',
        'They have a sturdy construction and are often used for storing and transporting liquids, such as wine or oil.',
        'Barrels can also be used for decorative purposes or as props in various industries, including winemaking and brewing.'
    ],
      '34': [
        'A hair slide, also called a hairpin or hair clip, is a small accessory used to secure or adorn hairstyles.',
        'It has a thin, elongated shape with a clasp or closure mechanism to hold the hair in place.',
        'Hair slides come in various designs, materials, and embellishments, allowing for personal style and creativity in hairstyling.'
    ],
    '33': [
        'A frying pan, also known as a skillet, is a shallow, flat-bottomed cooking utensil with sloping sides.',
        'It has a round or oval shape and is typically made of metal, such as stainless steel or cast iron.',
        'Frying pans are used for frying, searing, and sautéing food, and they often have a long handle for easy maneuvering.'
    ],
    '4': [
        'A harvestman, also known as a daddy longlegs, is a type of arachnid that resembles a spider.',
        'It has a small body with long, slender legs and lacks venom or silk-producing glands.',
        'Harvestmen are known for their scavenging behavior and are commonly found in gardens, forests, and other habitats.'
    ],
    '17': [
        'The Arctic fox is a small-sized fox species adapted to living in cold and snowy environments.',
        'It has a fluffy white or blue-gray coat that provides insulation in the Arctic regions.',
        'Arctic foxes have thick fur, a bushy tail, and rounded ears, and they change their coat color with the seasons.'
    ],
    '35': [
        'A Wrangler holster is a type of firearm holster designed specifically for Wrangler-style revolvers.',
        'It has a custom fit to securely hold the revolver and keep it accessible for quick draw and holstering.',
        'Wrangler holsters are often made of leather and feature belt loops or clips for attachment to a belt or waistband.'
    ],
    '60': [
        'An orange is a round citrus fruit with a bright orange-colored rind and juicy flesh inside.',
        'It has a spherical shape and is typically segmented into individual sections.',
        'Oranges are known for their sweet and tangy flavor and are rich in vitamin C.'
    ],
    '51': [
        'A tobacco shop, also known as a tobacconist, is a retail establishment that specializes in selling tobacco products.',
        'It may offer a variety of tobacco-related items, such as cigarettes, cigars, pipes, tobacco leaves, and smoking accessories.',
        'Tobacco shops often provide a wide selection of tobacco brands and cater to customers who enjoy smoking or collecting tobacco products.'
    ],
    '47': [
        'A spider web is a intricate structure created by spiders to capture prey.',
        'It has a delicate and intricate design, usually made of silk threads that are produced by the spider.',
        'Spider webs come in various shapes and sizes and are known for their strength and ability to trap insects.'
    ],
    '54': [
        'A wok is a versatile cooking pan with a rounded bottom and high, sloping sides.',
        'It has a circular or concave shape and is typically made of metal, such as carbon steel or cast iron.',
        'Woks are commonly used in Asian cuisine and are ideal for stir-frying, deep-frying, and other high-heat cooking methods.'
    ],
        '20': [
        'The rock beauty is a colorful species of fish found in coral reefs and rocky coastal areas.',
        'It has a compressed body with vibrant yellow, orange, and black vertical stripes.',
        'Rock beauties are known for their striking appearance and are popular among aquarium enthusiasts.'
    ],
    '56': [
        'A catamaran is a type of boat or watercraft with two parallel hulls connected by a deck or frame.',
        'It has a unique multihull design and is often used for sailing, cruising, or water sports.',
        'Catamarans offer stability, speed, and spaciousness, making them popular for recreational and commercial purposes.'
    ],
    '55': [
        'A worm fence, also known as a snake fence or zigzag fence, is a type of fence constructed from wooden rails.',
        'It has a distinctive zigzag or serpentine pattern, with the rails overlapping and interlocking.',
        'Worm fences are commonly used in rural and agricultural settings to enclose fields or mark boundaries.'
    ],
    '44': [
        'A slot is a narrow opening or groove, often found in machines or devices.',
        'It has a elongated and rectangular shape, allowing for the insertion of objects or components.',
        'Slots are used for various purposes, such as inserting coins, cards, or connectors.'
    ],
    '36': [
        'Lipstick is a cosmetic product used to add color, moisture, and definition to the lips.',
        'It has a cylindrical shape, typically housed in a tube with a twist mechanism for application.',
        'Lipsticks come in a wide range of shades and finishes, allowing for different lip looks and styles.'
    ],
    '38': [
        'An organ is a musical instrument that produces sound by air passing through pipes or reeds.',
        'It has a large and complex structure with multiple keyboards, pedals, and stops.',
        'Organs are known for their powerful and rich tones, and they are often found in churches or concert halls.'
    ],
    '15': [
        'The Newfoundland dog is a large and gentle breed known for its strength and swimming abilities.',
        'It has a massive, muscular body with a thick double coat that provides insulation in cold water.',
        'Newfoundland dogs are characterized by their kind and patient nature, making them excellent family pets and rescue dogs.'
    ],
    '48': [
        'A stage is a raised platform or area used for performances, presentations, or public speaking.',
        'It has a rectangular or circular shape, often equipped with lighting, sound systems, and props.',
        'Stages can be found in theaters, concert halls, auditoriums, and other venues where live events take place.'
    ],
    '40': [
        'A pencil box, also known as a pencil case or pencil holder, is a container used for storing and organizing writing and drawing instruments.',
        'It has a rectangular or cylindrical shape with compartments or slots to hold pencils, pens, erasers, and other stationery items.',
        'Pencil boxes come in various materials, such as plastic, metal, or fabric, and are commonly used by students and artists.'
    ],
     '9': [
        'The Saluki is a slender and graceful breed of dog known for its speed and agility.',
        'It has a slim body, a long and narrow head, and a silky coat that can come in different colors.',
        'Salukis are sight hounds and were historically used for hunting and coursing game in the deserts of the Middle East.'
    ],
    '37': [
        'The oboe is a woodwind instrument known for its distinct timbre and expressive capabilities.',
        'It has a cylindrical body with a double reed mouthpiece and multiple keys for fingerings.',
        'Oboes are commonly used in classical music and have a rich, vibrant sound that can evoke various emotions.'
    ]
}


enco_ddict ={}
model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()

input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

for key in dict_train.keys():
    list_temp=[]
    for dis in dict_train[key]:
        tt = clip.tokenize([dis]).cuda()
        text_features = model.encode_text(tt).float()
        list_temp.append(text_features.cpu().detach().numpy())
        
    enco_ddict[int(key)] = torch.tensor(np.mean(np.array(list_temp),axis=0))

path_img ="/home/ubuntu/Few_shot/few-shot-meta-baseline/all_logits.pickle"
path_label = "/home/ubuntu/Few_shot/few-shot-meta-baseline/all_labels.pickle"


cuda = True if torch.cuda.is_available() else False
print(cuda)

with open(path_img, 'rb') as f:
    data = pickle.load(f)
    feature = np.array(data)
    print(feature.shape)
          
with open(path_label, 'rb') as f:
    label = pickle.load(f)
    label = np.array(label)
    print(label.shape)
        
One_hot_encoding=[]
class_feature_means = {}

# Use a for loop to add keys from 1 to 63
class_size = 64
for i in range(class_size):
    class_feature_means[i] = []

j=0
for i in label:
    class_feature_means[i].append(feature[j])
    j=j+1

# checking wether the input visual fetures are correct using consine distance 
# print(1-torch.nn.functional.cosine_similarity(torch.tensor(feature[1]).unsqueeze(0),torch.tensor( feature[2000]).unsqueeze(0)))
# print(1-torch.nn.functional.cosine_similarity(torch.tensor(feature[1]).unsqueeze(0),torch.tensor( feature[2300]).unsqueeze(0)))
# print(1-torch.nn.functional.cosine_similarity(torch.tensor(feature[2]).unsqueeze(0),torch.tensor( feature[2600]).unsqueeze(0)))
# print(1-torch.nn.functional.cosine_similarity(torch.tensor(feature[1]).unsqueeze(0),torch.tensor( feature[2]).unsqueeze(0)))
# print(1-torch.nn.functional.cosine_similarity(torch.tensor(feature[1]).unsqueeze(0),torch.tensor( feature[20]).unsqueeze(0)))
# print(1-torch.nn.functional.cosine_similarity(torch.tensor(feature[800]).unsqueeze(0),torch.tensor( feature[820]).unsqueeze(0)))
# print(1-torch.nn.functional.cosine_similarity(torch.tensor(class_feature_means[1][2]).unsqueeze(0),torch.tensor( class_feature_means[1][2]).unsqueeze(0)))
# print(1-torch.nn.functional.cosine_similarity(torch.tensor(class_feature_means[1][2]).unsqueeze(0),torch.tensor( class_feature_means[1][3]).unsqueeze(0)))
# print(1-torch.nn.functional.cosine_similarity(torch.tensor(class_feature_means[1][2]).unsqueeze(0),torch.tensor( class_feature_means[1][4]).unsqueeze(0)))
# print(1-torch.nn.functional.cosine_similarity(torch.tensor(class_feature_means[1][2]).unsqueeze(0),torch.tensor( class_feature_means[1][5]).unsqueeze(0)))
# print(1-torch.nn.functional.cosine_similarity(torch.tensor(class_feature_means[1][2]).unsqueeze(0),torch.tensor( class_feature_means[1][6]).unsqueeze(0)))

for i in range(class_size):
    class_feature_means[int(i)] = torch.tensor(np.mean(np.array(class_feature_means[i]),axis=0))
# print("wwe1")

for i in label:
  a=np.zeros(class_size)
  a[int(i)]=1
  One_hot_encoding.append(torch.reshape(torch.from_numpy(a),(1,class_size)))

label=torch.cat(One_hot_encoding,dim=0)
fe_tensor = torch.from_numpy(feature)
lb_tensor = label
features,fe_label = shuffle(fe_tensor.numpy(), lb_tensor.numpy(), random_state=0)
print(features.shape,fe_label.shape)


BUFFER_SIZE =38400
BATCH_SIZE = 4800
NOISE_DIM = 512
BETA1 = 0.5
BETA2 = 0.999
noise_dim =512
class_size = 64


checkpoint_interval = 5
output_dir = "rfs_text_checkpoints_mini_visual_3"
latest_checkpoint = None
checkpoint_epoch = 0


loader = DataLoader(list(zip(torch.tensor(features), torch.tensor(fe_label))), shuffle=True, batch_size=BATCH_SIZE)


# Define models
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 8192),
            nn.BatchNorm1d(8192),
            nn.LeakyReLU(0.2),
            nn.Linear(8192,4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
       x=self.model(x)
       #x = normalize(x, p=1, dim = 1)   
       return x 


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(NOISE_DIM, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,class_size),
            nn.BatchNorm1d(class_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)
        

#loss functions
cross_entropy =  torch.nn.BCELoss()


def create_class_feature_batch(label_batch, dictionary):
    batch_size = label_batch.size(0)
    feature_size = next(iter(dictionary.values())).size(0)  

    class_features = []
    for label in label_batch:
        class_feature = dictionary.get(int(label), None)
        if class_feature is not None:
            class_features.append(class_feature)

    if len(class_features) > 0:
        class_feature_batch = torch.stack(class_features)
    else:
        class_feature_batch = torch.empty((0, feature_size))

    return class_feature_batch


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(real_output,torch.ones_like(real_output)).clone()
    fake_loss = cross_entropy(fake_output,torch.zeros_like(fake_output)).clone()
    total_loss = real_loss + fake_loss 
    return (total_loss)


def classfier_loss(y_true,y_pred):
    return cross_entropy( y_pred.double(),y_true.double())


def generator_loss(fake_out,y_true,y_pred):
    gen_discrminator_loss= cross_entropy(y_pred.double(),y_true.double())
    gen_classfier_loss= cross_entropy( fake_out,torch.ones_like(fake_out))
    return gen_discrminator_loss+gen_classfier_loss


device = torch.device('cuda')


# Initialize models and optimizers
generator = Generator().to('cuda')
discriminator = Discriminator().to('cuda')
classifier = Classifier().to('cuda')
generator_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(BETA1,BETA2))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(BETA1,BETA2))
classifier_optimizer = optim.Adam(classifier.parameters(), lr=4e-4, betas=(BETA1,BETA2))


# if stopped load your checkpoint here
if latest_checkpoint:
    # Load the latest checkpoint
    generator.load_state_dict(torch.load("/home/heethanjan/Heetha/other/text_checkpoints_tire_visual/epoch_4615/generator.pth", map_location=device))
    discriminator.load_state_dict(torch.load("/home/heethanjan/Heetha/other/text_checkpoints_tire_visual/epoch_4615/discriminator.pth", map_location=device))
    classifier.load_state_dict(torch.load("/home/heethanjan/Heetha/other/text_checkpoints_tire_visual/epoch_4615/classifier.pth", map_location=device))

    generator_optimizer.load_state_dict(torch.load("/home/heethanjan/Heetha/other/text_checkpoints_tire_visual/epoch_4615/generator_optimizer.pth"))
    discriminator_optimizer.load_state_dict(torch.load("/home/heethanjan/Heetha/other/text_checkpoints_tire_visual/epoch_4615/discriminator_optimizer.pth"))
    classifier_optimizer.load_state_dict(torch.load("/home/heethanjan/Heetha/other/text_checkpoints_tire_visual/epoch_4615/classifier_optimizer.pth"))
    checkpoint_epoch
    print(f"Loaded checkpoint from epoch {checkpoint_epoch}")


def get_batch_features(labels, feature_dict):
    batch_features = []
    for label in labels:
        feature = feature_dict[label.item()]
        batch_features.append(feature.reshape((512,)))   
    batch_tensor = torch.stack(batch_features)
    return batch_tensor


def convert_one_hot_to_class(batch_labels):
    class_indices = torch.argmax(batch_labels, dim=1)
    return class_indices


def train_step(images, one_hot_labels):
    labels_as_class = convert_one_hot_to_class(one_hot_labels)
    text_features = get_batch_features(labels_as_class,enco_ddict)
    text_features = text_features.to('cuda')
   
    # Generate images and calculate loss values
    
    # Classifier Loss 
    classifier_optimizer.zero_grad()
    real_classifier = classifier(images.to('cuda').to(torch.float32))
    clss_loss = classfier_loss(one_hot_labels.to('cuda'), real_classifier.to('cuda'))
    clss_loss.backward()
    classifier_optimizer.step()

    # Discriminator Loss
    discriminator_optimizer.zero_grad()
    text_features = text_features.to('cuda')
    generated_images = generator(text_features.to('cuda'))
    gen_classifier = classifier(generated_images.to('cuda').to(torch.float32))
    real_classifier = classifier(images.to('cuda').to(torch.float32))
    real_output = discriminator(images.to('cuda').to(torch.float32))
    fake_output = discriminator(generated_images.to('cuda').to(torch.float32))
    disc_loss = discriminator_loss(real_output, fake_output).to('cuda')
    disc_loss.backward()
    discriminator_optimizer.step()

    # Generator Loss
    generator_optimizer.zero_grad()
    text_features = text_features.to('cuda')
    generated_images = generator(text_features.to('cuda'))
    
    gen_classifier = classifier(generated_images.to('cuda').to(torch.float32))
    real_classifier = classifier(images.to('cuda').to(torch.float32))
    
    real_output = discriminator(images.to('cuda').to(torch.float32))
    fake_output = discriminator(generated_images.to('cuda').to(torch.float32))
    
    class_feature_batch = create_class_feature_batch(labels_as_class, class_feature_means)
    mean_same_class_distance = (1-torch.abs(torch.nn.functional.cosine_similarity(class_feature_batch.to('cuda'),generated_images.to('cuda'),dim=1))).mean()
    gen_loss_1 = generator_loss(fake_output.to('cuda'), one_hot_labels.to('cuda'), gen_classifier.to('cuda'))
    gen_loss = gen_loss_1 + mean_same_class_distance
    gen_loss.backward()
    generator_optimizer.step()


def train(loader,epochs):
    for epoch in range(checkpoint_epoch,epochs):
        start = time.time()
        for image_batch, label_batch in loader:
            train_step(image_batch, label_batch)
        
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        if (epoch + 1) % checkpoint_interval == 0:
            # Create a new directory for this checkpoint if it doesn't exist
            checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save generator and discriminator checkpoints
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, "generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, "discriminator.pth"))
            torch.save(classifier.state_dict(), os.path.join(checkpoint_dir, "classifier.pth"))

            torch.save(generator_optimizer.state_dict(), os.path.join(checkpoint_dir, "generator_optimizer.pth"))
            torch.save(discriminator_optimizer.state_dict(), os.path.join(checkpoint_dir, "discriminator_optimizer.pth"))
            torch.save(classifier_optimizer.state_dict(), os.path.join(checkpoint_dir, "classifier_optimizer.pth"))
            # Delete old checkpoints
            if epoch >= checkpoint_interval:
                old_checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch + 1 - checkpoint_interval}")
                os.system(f"rm -rf {old_checkpoint_dir}")
train(loader,1000)
model = generator
torch.save(model.state_dict(), "text-mini-gen-model.pth")