import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance
from torch.autograd import Variable
import pickle
import clip
from torch.nn.functional import normalize
import torch
import torch.nn as nn

# Define the Generator model class
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
        x = self.model(x)
        return x

# Load the generator model
generator = Generator().to('cuda')

# Load the pre-trained weights
generator.load_state_dict(torch.load("/home/ubuntu/Few_shot/new_no_discriminator_w-1-text-mini-gen-model.pth"))

# Set the model to evaluation mode
generator.eval()

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

test_dic = {
    '80': [
        'The Dalmatian is a medium-sized dog breed known for its distinctive coat pattern of black spots on a white background.',
        'It has a muscular body, a slightly rounded head, and alert, expressive eyes.',
        'Dalmatians are known for their energetic and playful nature, making them popular as family pets and firehouse mascots.'
    ],
    '81': [
        'Nematodes are a diverse group of worms that can be found in various habitats, including soil, freshwater, and marine environments.',
        'They have a slender, unsegmented body with a tapered shape at both ends.',
        'Nematodes play important roles in ecosystems as decomposers, predators, and parasites.'
    ],
    '82': [
        'An ant is a small insect known for its organized social behavior and strong sense of colony cooperation.',
        'It has a segmented body with six legs and a pair of antennae on its head.',
        'Ants are highly adaptable and can be found in diverse habitats, forming intricate underground nests or building mounds.'
    ],
    '83': [
        'The black-footed ferret is a small carnivorous mammal and one of the most endangered species in North America.',
        'It has a slender body, a long neck, and a distinctive black mask on its face.',
        'Black-footed ferrets are known for their agility and hunting skills, primarily preying on prairie dogs.'
    ],
    '84': [
        'The king crab is a large crustacean known for its impressive size and spiky appearance.',
        'It has a hard exoskeleton, long legs, and large pincers used for feeding and defense.',
        'King crabs are prized for their meat and are popular in seafood cuisine.'
    ],
    '85': [
        'The lion is a majestic big cat known for its golden mane and powerful build.',
        'It has a large, muscular body, a rounded head, and a roar that can be heard from a considerable distance.',
        'Lions are apex predators and live in social groups known as prides.'
    ],
    '86': [
        'A vase is a decorative container usually made of glass, ceramic, or porcelain.',
        'It has a cylindrical or bulbous shape with a narrow neck and is used for holding flowers or as a standalone decorative piece.',
        'Vases come in various sizes, colors, and designs, adding beauty and elegance to interior spaces.'
    ],
    '87': [
        'The golden retriever is a friendly and intelligent dog breed with a dense, water-repellent coat.',
        'It has a sturdy body, a broad head, and a wagging tail that showcases its joyful nature.',
        'Golden retrievers are known for their loyalty and versatility, excelling as family pets, guide dogs, and search-and-rescue companions.'
    ],
    '88': [
        'A mixing bowl is a kitchen utensil used for combining ingredients during food preparation.',
        'It has a rounded or cylindrical shape with a wide opening and a flat base.',
        'Mixing bowls come in various sizes and materials, such as stainless steel or ceramic, and are essential for baking and cooking.'
    ],
        '89': [
        'The Malamute is a large and powerful dog breed known for its strength and endurance.',
        'It has a sturdy body, a thick double coat, and erect ears.',
        'Malamutes are sled dogs and have a friendly and sociable temperament.'
    ],
    '90': [
        'The African Hunting Dog, also known as the African Wild Dog or Painted Wolf, is a highly social and skilled predator.',
        'It has a lean body, large rounded ears, and a unique coat pattern with patches of black, white, and tan.',
        'African Hunting Dogs live in packs and are known for their cooperative hunting style and impressive stamina.'
    ],
    '91': [
        'A cuirass is a piece of armor that protects the torso and consists of a breastplate and a backplate.',
        'It has a flat and rigid structure, providing protection to the chest and abdomen.',
        'Cuirasses were commonly used by ancient warriors and knights in combat.'
    ],
    '92': [
        'A bookshop, also known as a bookstore, is a retail establishment where books are sold.',
        'It typically has shelves or racks filled with books of various genres and subjects.',
        'Bookshops often provide a cozy and inviting atmosphere for browsing and discovering new literary treasures.'
    ],
    '93': [
        'A crate is a sturdy container made of wood, plastic, or metal, used for storing or transporting goods.',
        'It has a rectangular shape with solid walls and often features handles or latches for ease of handling.',
        'Crates are versatile and commonly used in logistics, storage, and pet training.'
    ],
    '94': [
        'An hourglass is a timekeeping device consisting of two glass bulbs connected by a narrow neck, with sand or granules flowing from one bulb to the other.',
        'It has a distinct hourglass shape, with the upper and lower bulbs symmetrically aligned.',
        'Hourglasses were historically used to measure time and are often associated with concepts of passing time and urgency.'
    ],
    '95': [
        'The electric guitar is a stringed musical instrument that uses electromagnetic pickups to convert string vibrations into electrical signals.',
        'It has a solid or semi-hollow body, a neck with frets, and multiple pickups and controls for tone and volume adjustment.',
        'Electric guitars are widely used in various music genres, offering a wide range of sounds and styles.'
    ],
    '96': [
        'A trifle is a layered dessert typically made with sponge cake, custard, fruit, and whipped cream.',
        'It is usually presented in a glass bowl or dish, showcasing its visually appealing layers.',
        'Trifles are popular desserts for special occasions and gatherings, offering a delightful combination of textures and flavors.'
    ],
    '97': [
        'A school bus is a type of vehicle designed to transport students to and from school.',
        'It has a large, rectangular body with multiple rows of seats, safety features, and distinctive yellow paint.',
        'School buses are essential for safe and reliable transportation of students.'
    ],
    '98': [
        'A theater curtain, also known as a stage curtain or drape, is a large piece of fabric used to conceal or reveal the stage during theatrical performances.',
        'It can be made of various materials, such as velvet or heavy fabric, and is often adorned with decorative elements.',
        'Theater curtains add drama and anticipation, enhancing the visual experience of the audience.'
    ],
    '99': [
        'A scoreboard is an electronic or manual display used to keep track of scores or other important information during sports or competitive events.',
        'It typically consists of a series of panels or digits that can be updated to show numbers or text.',
        'Scoreboards are essential for providing real-time information to participants and spectators.'
    ]
}

enco_ddict ={}
test_enco_ddict ={}
model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

for key in test_dic.keys():
    list_temp=[]
    for dis in test_dic[key]:
        tt = clip.tokenize([dis]).cuda()
        text_features = model.encode_text(tt).float()
        list_temp.append(text_features.cpu().detach().numpy())
    #print(len(list_temp))
    test_enco_ddict[int(key)-80] = torch.tensor(np.mean(np.array(list_temp),axis=0))


for key in dict_train.keys():
    list_temp=[]
    for dis in dict_train[key]:
        tt = clip.tokenize([dis]).cuda()
        text_features = model.encode_text(tt).float()
        list_temp.append(text_features.cpu().detach().numpy())
    #print(len(list_temp))
    enco_ddict[int(key)] = torch.tensor(np.mean(np.array(list_temp),axis=0))


@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query,x_shot_label):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]
        lab_shape = x_shot_label.shape[-1:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_shot_label = x_shot_label.view(-1, *lab_shape)
        text_features_list = []
        for label in x_shot_label:
            label_class = int(label.item())  # Get the class label as an integer
            text_feature = test_enco_ddict[label_class]  # Get the corresponding text feature from the dictionary
            text_features_list.append(text_feature)

# Stack the text features to create a single tensor
        stacked_text_features = torch.stack(text_features_list, dim=0)
        stacked_text_features = stacked_text_features.squeeze(dim=1)
        #print(stacked_text_features.shape)
# Now, you can pass the stacked text features to the generator
        generated_features = generator(stacked_text_features.to('cuda'))

        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)
        x_shot_label  = x_shot_label.view(*shot_shape, -1)
        x_shot_gen = generated_features.view(*shot_shape, -1)

        #print(x_shot_gen.shape,generated_features.shape)
        # x_shot = F.normalize(x_shot, dim=-1)
        # x_shot_gen = F.normalize(x_shot_gen, dim=-1)
        x_shot = (x_shot + x_shot_gen) / 2
       
        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        return logits

