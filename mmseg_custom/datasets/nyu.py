# from .builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import DATASETS

@DATASETS.register_module()
class NyuDataset(CustomDataset):
    CLASSES = ('book', 'bottle', 'cabinet', 'ceiling', 'chair', 'cone', 'counter', 'dishwasher', 'faucet', 'fire extinguisher', 'floor', 'garbage bin', 'microwave', 'paper towel dispenser', 'paper', 'pot', 'refridgerator', 'stove burner', 'table', 'unknown', 'wall', 'bowl', 'magnet', 'sink', 'air vent', 'box', 'door knob', 'door', 'scissor', 'tape dispenser', 'telephone cord', 'telephone', 'track light', 'cork board', 'cup', 'desk', 'laptop', 'air duct', 'basket', 'camera', 'pipe', 'shelves', 'stacked chairs', 'styrofoam object', 'whiteboard', 'computer', 'keyboard', 'ladder', 'monitor', 'stand', 'bar', 'motion camera', 'projector screen', 'speaker', 'bag', 'clock', 'green screen', 'mantel', 'window', 'ball', 'hole puncher', 'light', 'manilla envelope', 'picture', 'mail shelf', 'printer', 'stapler', 'fax machine', 'folder', 'jar', 'magazine', 'ruler', 'cable modem', 'fan', 'file', 'hand sanitizer', 'paper rack', 'vase', 'air conditioner', 'blinds', 'flower', 'plant', 'sofa', 'stereo', 'books', 'exit sign', 'room divider', 'bookshelf', 'curtain', 'projector', 'modem', 'wire', 'water purifier', 'column', 'hooks', 'hanging hooks', 'pen', 'electrical outlet', 'doll', 'eraser', 'pencil holder', 'water carboy', 'mouse', 'cable rack', 'wire rack', 'flipboard', 'map', 'paper cutter', 'tape', 'thermostat', 'heater', 'circuit breaker box', 'paper towel', 'stamp', 'duster', 'poster case', 'whiteboard marker', 'ethernet jack', 'pillow', 'hair brush', 'makeup brush', 'mirror', 'shower curtain', 'toilet', 'toiletries bag', 'toothbrush holder', 'toothbrush', 'toothpaste', 'platter', 'rug', 'squeeze tube', 'shower cap', 'soap', 'towel rod', 'towel', 'bathtub', 'candle', 'tissue box', 'toilet paper', 'container', 'clothes', 'electric toothbrush', 'floor mat', 'lamp', 'drum', 'flower pot', 'banana', 'candlestick', 'shoe', 'stool', 'urn', 'earplugs', 'mailshelf', 'placemat', 'excercise ball', 'alarm clock', 'bed', 'night stand', 'deoderant', 'headphones', 'headboard', 'basketball hoop', 'foot rest', 'laundry basket', 'sock', 'football', 'mens suit', 'cable box', 'dresser', 'dvd player', 'shaver', 'television', 'contact lens solution bottle', 'drawer', 'remote control', 'cologne', 'stuffed animal', 'lint roller', 'tray', 'lock', 'purse', 'toy bottle', 'crate', 'vasoline', 'gift wrapping roll', 'wall decoration', 'hookah', 'radio', 'bicycle', 'pen box', 'mask', 'shorts', 'hat', 'hockey glove', 'hockey stick', 'vuvuzela', 'dvd', 'chessboard', 'suitcase', 'calculator', 'flashcard', 'staple remover', 'umbrella', 'bench', 'yoga mat', 'backpack', 'cd', 'sign', 'hangers', 'notebook', 'hanger', 'security camera', 'folders', 'clothing hanger', 'stairs', 'glass rack', 'saucer', 'tag', 'dolly', 'machine', 'trolly', 'shopping baskets', 'gate', 'bookrack', 'blackboard', 'coffee bag', 'coffee packet', 'hot water heater', 'muffins', 'napkin dispenser', 'plaque', 'plastic tub', 'plate', 'coffee machine', 'napkin holder', 'radiator', 'coffee grinder', 'oven', 'plant pot', 'scarf', 'spice rack', 'stove', 'tea kettle', 'napkin', 'bag of chips', 'bread', 'cutting board', 'dish brush', 'serving spoon', 'sponge', 'toaster', 'cooking pan', 'kitchen items', 'ladel', 'spatula', 'spice stand', 'trivet', 'knife rack', 'knife', 'baking dish', 'dish scrubber', 'drying rack', 'vessel', 'kichen towel', 'tin foil', 'kitchen utensil', 'utensil', 'blender', 'garbage bag', 'sink protector', 'box of ziplock bags', 'spice bottle', 'pitcher', 'pizza box', 'toaster oven', 'step stool', 'vegetable peeler', 'washing machine', 'can opener', 'can of food', 'paper towel holder', 'spoon stand', 'spoon', 'wooden kitchen utensils', 'bag of flour', 'fruit', 'sheet of metal', 'waffle maker', 'cake', 'cell phone', 'tv stand', 'tablecloth', 'wine glass', 'sculpture', 'wall stand', 'iphone', 'coke bottle', 'piano', 'wine rack', 'guitar', 'light switch', 'shirts in hanger', 'router', 'glass pot', 'cart', 'vacuum cleaner', 'bin', 'coins', 'hand sculpture', 'ipod', 'jersey', 'blanket', 'ironing board', 'pen stand', 'mens tie', 'glass baking dish', 'utensils', 'frying pan', 'shopping cart', 'plastic bowl', 'wooden container', 'onion', 'potato', 'jacket', 'dvds', 'surge protector', 'tumbler', 'broom', 'can', 'crock pot', 'person', 'salt shaker', 'wine bottle', 'apple', 'eye glasses', 'menorah', 'bicycle helmet', 'fire alarm', 'water fountain', 'humidifier', 'necklace', 'chandelier', 'barrel', 'chest', 'decanter', 'wooden utensils', 'globe', 'sheets', 'fork', 'napkin ring', 'gift wrapping', 'bed sheets', 'spot light', 'lighting track', 'cannister', 'coffee table', 'mortar and pestle', 'stack of plates', 'ottoman', 'server', 'salt container', 'utensil container', 'phone jack', 'switchbox', 'casserole dish', 'oven handle', 'whisk', 'dish cover', 'electric mixer', 'decorative platter', 'drawer handle', 'fireplace', 'stroller', 'bookend', 'table runner', 'typewriter', 'ashtray', 'key', 'suit jacket', 'range hood', 'cleaning wipes', 'six pack of beer', 'decorative plate', 'watch', 'balloon', 'ipad', 'coaster', 'whiteboard eraser', 'toy', 'toys basket', 'toy truck', 'classroom board', 'chart stand', 'picture of fish', 'plastic box', 'pencil', 'carton', 'walkie talkie', 'binder', 'coat hanger', 'filing shelves', 'plastic crate', 'plastic rack', 'plastic tray', 'flag', 'poster board', 'lunch bag', 'board', 'leg of a girl', 'file holder', 'chart', 'glass pane', 'cardboard tube', 'bassinet', 'toy car', 'toy shelf', 'toy bin', 'toys shelf', 'educational display', 'placard', 'soft toy group', 'soft toy', 'toy cube', 'toy cylinder', 'toy rectangle', 'toy triangle', 'bucket', 'chalkboard', 'game table', 'storage shelvesbooks', 'toy cuboid', 'toy tree', 'wooden toy', 'toy box', 'toy phone', 'toy sink', 'toyhouse', 'notecards', 'toy trucks', 'wall hand sanitizer dispenser', 'cap stand', 'music stereo', 'toys rack', 'display board', 'lid of jar', 'stacked bins  boxes', 'stacked plastic racks', 'storage rack', 'roll of paper towels', 'cables', 'power surge', 'cardboard sheet', 'banister', 'show piece', 'pepper shaker', 'kitchen island', 'excercise equipment', 'treadmill', 'ornamental plant', 'piano bench', 'sheet music', 'grandfather clock', 'iron grill', 'pen holder', 'toy doll', 'globe stand', 'telescope', 'magazine holder', 'file container', 'paper holder', 'flower box', 'pyramid', 'desk mat', 'cordless phone', 'desk drawer', 'envelope', 'window frame', 'id card', 'file stand', 'paper weight', 'toy plane', 'money', 'papers', 'comforter', 'crib', 'doll house', 'toy chair', 'toy sofa', 'plastic chair', 'toy house', 'child carrier', 'cloth bag', 'cradle', 'baby chair', 'chart roll', 'toys box', 'railing', 'clothing dryer', 'clothing washer', 'laundry detergent jug', 'clothing detergent', 'bottle of soap', 'box of paper', 'trolley', 'hand sanitizer dispenser', 'soap holder', 'water dispenser', 'photo', 'water cooler', 'foosball table', 'crayon', 'hoola hoop', 'horse toy', 'plastic toy container', 'pool table', 'game system', 'pool sticks', 'console system', 'video game', 'pool ball', 'trampoline', 'tricycle', 'wii', 'furniture', 'alarm', 'toy table', 'ornamental item', 'copper vessel', 'stick', 'car', 'mezuza', 'toy cash register', 'lid', 'paper bundle', 'business cards', 'clipboard', 'flatbed scanner', 'paper tray', 'mouse pad', 'display case', 'tree sculpture', 'basketball', 'fiberglass case', 'framed certificate', 'cordless telephone', 'shofar', 'trophy', 'cleaner', 'cloth drying stand', 'electric box', 'furnace', 'piece of wood', 'wooden pillar', 'drying stand', 'cane', 'clothing drying rack', 'iron box', 'excercise machine', 'sheet', 'rope', 'sticks', 'wooden planks', 'toilet plunger', 'bar of soap', 'toilet bowl brush', 'light bulb', 'drain', 'faucet handle', 'nailclipper', 'shaving cream', 'rolled carpet', 'clothing iron', 'window cover', 'charger and wire', 'quilt', 'mattress', 'hair dryer', 'stones', 'pepper grinder', 'cat cage', 'dish rack', 'curtain rod', 'calendar', 'head phones', 'cd disc', 'head phone', 'usb drive', 'water heater', 'pan', 'tuna cans', 'baby gate', 'spoon sets', 'cans of cat food', 'cat', 'flower basket', 'fruit platter', 'grapefruit', 'kiwi', 'hand blender', 'knobs', 'vessels', 'cell phone charger', 'wire basket', 'tub of tupperware', 'candelabra', 'litter box', 'shovel', 'cat bed', 'door way', 'belt', 'surge protect', 'glass', 'console controller', 'shoe rack', 'door frame', 'computer disk', 'briefcase', 'mail tray', 'file pad', 'letter stand', 'plastic cup of coffee', 'glass box', 'ping pong ball', 'ping pong racket', 'ping pong table', 'tennis racket', 'ping pong racquet', 'xbox', 'electric toothbrush base', 'toilet brush', 'toiletries', 'razor', 'bottle of contact lens solution', 'contact lens case', 'cream', 'glass container', 'container of skin cream', 'soap dish', 'scale', 'soap stand', 'cactus', 'door  window  reflection', 'ceramic frog', 'incense candle', 'storage space', 'door lock', 'toilet paper holder', 'tissue', 'personal care liquid', 'shower head', 'shower knob', 'knob', 'cream tube', 'perfume box', 'perfume', 'back scrubber', 'door facing trimreflection', 'doorreflection', 'light switchreflection', 'medicine tube', 'wallet', 'soap tray', 'door curtain', 'shower pipe', 'face wash cream', 'flashlight', 'shower base', 'window shelf', 'shower hose', 'toothpaste holder', 'soap box', 'incense holder', 'conch shell', 'roll of toilet paper', 'shower tube', 'bottle of listerine', 'bottle of hand wash liquid', 'tea pot', 'lazy susan', 'avocado', 'fruit stand', 'fruitplate', 'oil container', 'package of water', 'bottle of liquid', 'door way arch', 'jug', 'bulb', 'bagel', 'bag of bagels', 'banana peel', 'bag of oreo', 'flask', 'collander', 'brick', 'torch', 'dog bowl', 'wooden plank', 'eggs', 'grill', 'dog', 'chimney', 'dog cage', 'orange plastic cap', 'glass set', 'vessel set', 'mellon', 'aluminium foil', 'orange', 'peach', 'tea coaster', 'butterfly sculpture', 'corkscrew', 'heating tray', 'food processor', 'corn', 'squash', 'watermellon', 'vegetables', 'celery', 'glass dish', 'hot dogs', 'plastic dish', 'vegetable', 'sticker', 'chapstick', 'sifter', 'fruit basket', 'glove', 'measuring cup', 'water filter', 'wine accessory', 'dishes', 'file box', 'ornamental pot', 'dog toy', 'salt and pepper', 'electrical kettle', 'kitchen container plastic', 'pineapple', 'suger jar', 'steamer', 'charger', 'mug holder', 'orange juicer', 'juicer', 'bag of hot dog buns', 'hamburger bun', 'mug hanger', 'bottle of ketchup', 'toy kitchen', 'food wrapped on a tray', 'kitchen utensils', 'oven mitt', 'bottle of comet', 'wooden utensil', 'decorative dish', 'handle', 'label', 'flask set', 'cooking pot cover', 'tupperware', 'garlic', 'tissue roll', 'lemon', 'wine', 'decorative bottle', 'wire tray', 'tea cannister', 'clothing hamper', 'guitar case', 'wardrobe', 'boomerang', 'button', 'karate belts', 'medal', 'window seat', 'window box', 'necklace holder', 'beeper', 'webcam', 'fish tank', 'luggage', 'life jacket', 'shoelace', 'pen cup', 'eyeball plastic ball', 'toy pyramid', 'model boat', 'certificate', 'puppy toy', 'wire board', 'quill', 'canister', 'toy boat', 'antenna', 'bean bag', 'lint comb', 'travel bag', 'wall divider', 'toy chest', 'headband', 'luggage rack', 'bunk bed', 'lego', 'yarmulka', 'package of bedroom sheets', 'bedding package', 'comb', 'dollar bill', 'pig', 'storage bin', 'storage chest', 'slide', 'playpen', 'electronic drumset', 'ipod dock', 'microphone', 'music keyboard', 'music stand', 'microphone stand', 'album', 'kinect', 'inkwell', 'baseball', 'decorative bowl', 'book holder', 'toy horse', 'desser', 'toy apple', 'toy dog', 'scenary', 'drawer knob', 'shoe hanger', 'tent', 'figurine', 'soccer ball', 'hand weight', 'magic 8ball', 'bottle of perfume', 'sleeping bag', 'decoration item', 'envelopes', 'trinket', 'hand fan', 'sculpture of the chrysler building', 'sculpture of the eiffel tower', 'sculpture of the empire state building', 'jeans', 'garage door', 'case', 'rags', 'decorative item', 'toy stroller', 'shelf frame', 'cat house', 'can of beer', 'dog bed', 'lamp shade', 'bracelet', 'reflection of window shutters', 'decorative egg', 'indoor fountain', 'photo album', 'decorative candle', 'walkietalkie', 'serving dish', 'floor trim', 'mini display platform', 'american flag', 'vhs tapes', 'throw', 'newspapers', 'mantle', 'package of bottled water', 'serving platter', 'display platter', 'centerpiece', 'tea box', 'gold piece', 'wreathe', 'lectern', 'hammer', 'matchbox', 'pepper', 'yellow pepper', 'duck', 'eggplant', 'glass ware', 'sewing machine', 'rolled up rug', 'doily', 'coffee pot', 'torah', 'background')
    
    PALETTE = [[119, 123, 220], [137, 115, 23], [102, 255, 255], [30, 154, 46], [25, 240, 49], [51, 83, 186], [69, 3, 5], [7, 52, 56], [116, 150, 9], [165, 2, 83], [72, 237, 160], [213, 26, 39], [4, 8, 98], [179, 111, 197], [158, 168, 218], [19, 121, 178], [210, 9, 91], [163, 60, 17], [18, 61, 85], [139, 62, 97], [35, 109, 1], [202, 201, 176], [114, 207, 130], [139, 31, 90], [130, 235, 154], [166, 70, 56], [169, 4, 123], [116, 78, 81], [199, 191, 236], [141, 165, 190], [38, 89, 15], [162, 135, 61], [49, 14, 10], [107, 204, 5], [68, 22, 219], [50, 215, 79], [230, 136, 124], [198, 172, 222], [78, 64, 29], [13, 14, 218], [193, 2, 75], [253, 239, 95], [134, 111, 125], [163, 187, 169], [39, 17, 56], [172, 82, 142], [9, 104, 51], [65, 246, 29], [255, 155, 108], [159, 76, 224], [65, 134, 211], [150, 95, 204], [152, 84, 12], [126, 170, 168], [36, 107, 240], [70, 72, 55], [155, 113, 32], [121, 124, 47], [88, 191, 243], [189, 199, 251], [208, 242, 119], [191, 105, 25], [14, 85, 157], [83, 88, 35], [34, 242, 19], [36, 230, 157], [43, 60, 184], [1, 12, 83], [74, 72, 65], [92, 95, 145], [3, 184, 174], [148, 138, 42], [203, 45, 202], [187, 121, 121], [22, 97, 248], [226, 134, 214], [149, 159, 164], [228, 166, 117], [211, 14, 252], [195, 68, 166], [255, 135, 23], [246, 184, 217], [53, 249, 67], [102, 113, 240], [105, 95, 96], [99, 141, 157], [84, 24, 14], [162, 188, 226], [203, 123, 214], [192, 154, 230], [29, 41, 48], [46, 225, 154], [148, 77, 85], [127, 123, 140], [180, 117, 143], [204, 155, 132], [192, 76, 190], [51, 55, 55], [202, 132, 3], [165, 206, 107], [6, 65, 53], [27, 51, 150], [203, 147, 101], [142, 8, 152], [14, 84, 223], [83, 121, 155], [28, 25, 121], [14, 181, 201], [49, 46, 69], [250, 213, 192], [113, 110, 243], [41, 44, 234], [44, 174, 243], [173, 214, 243], [79, 145, 182], [171, 168, 66], [88, 25, 170], [148, 131, 142], [165, 89, 237], [6, 164, 191], [60, 172, 189], [195, 171, 55], [120, 212, 5], [238, 164, 40], [188, 140, 220], [6, 167, 80], [69, 73, 186], [217, 25, 47], [248, 32, 46], [9, 72, 231], [167, 36, 23], [139, 242, 25], [38, 0, 106], [133, 205, 115], [233, 134, 232], [198, 52, 134], [146, 230, 143], [60, 43, 131], [170, 164, 220], [60, 171, 211], [17, 150, 182], [16, 90, 31], [73, 68, 185], [33, 168, 179], [80, 177, 12], [50, 165, 172], [199, 175, 182], [217, 137, 180], [93, 212, 216], [168, 67, 217], [25, 108, 35], [86, 6, 7], [16, 108, 131], [131, 69, 182], [244, 49, 207], [103, 124, 153], [0, 126, 215], [72, 55, 8], [17, 59, 217], [235, 65, 148], [205, 44, 36], [75, 211, 83], [206, 85, 222], [164, 68, 223], [22, 240, 122], [24, 101, 202], [133, 244, 223], [127, 122, 174], [140, 35, 40], [242, 114, 14], [187, 159, 77], [47, 23, 162], [229, 17, 16], [143, 73, 139], [134, 18, 192], [240, 246, 88], [10, 4, 206], [184, 96, 9], [153, 88, 197], [226, 134, 158], [43, 49, 198], [139, 179, 166], [113, 246, 101], [167, 127, 22], [82, 195, 120], [140, 48, 130], [20, 102, 184], [43, 47, 18], [186, 116, 151], [101, 12, 119], [244, 155, 248], [23, 24, 135], [128, 73, 237], [22, 32, 126], [102, 68, 135], [97, 82, 255], [207, 26, 51], [252, 186, 251], [17, 75, 101], [39, 133, 60], [201, 226, 44], [170, 252, 32], [23, 169, 241], [63, 25, 79], [212, 157, 174], [216, 81, 6], [64, 42, 140], [200, 191, 158], [196, 223, 191], [57, 66, 66], [98, 112, 106], [90, 99, 181], [213, 249, 110], [135, 29, 155], [121, 130, 154], [14, 40, 26], [240, 194, 186], [64, 117, 244], [205, 78, 168], [149, 145, 39], [153, 59, 30], [189, 200, 241], [118, 163, 22], [6, 253, 174], [81, 183, 171], [156, 138, 142], [113, 106, 180], [245, 87, 221], [247, 170, 171], [117, 250, 220], [68, 93, 126], [136, 189, 253], [255, 124, 61], [181, 105, 177], [51, 199, 156], [140, 242, 2], [85, 216, 20], [154, 82, 192], [31, 52, 249], [171, 178, 54], [236, 192, 204], [23, 6, 32], [162, 158, 166], [166, 200, 104], [62, 91, 32], [75, 105, 199], [166, 48, 63], [5, 64, 82], [175, 252, 13], [11, 217, 124], [109, 96, 217], [132, 240, 193], [149, 104, 112], [255, 255, 90], [250, 130, 161], [115, 104, 125], [103, 192, 248], [44, 202, 150], [152, 202, 250], [149, 103, 208], [208, 23, 167], [147, 171, 53], [128, 23, 36], [150, 19, 115], [75, 64, 8], [184, 169, 117], [80, 228, 169], [13, 163, 108], [191, 213, 236], [223, 153, 147], [70, 84, 160], [120, 81, 65], [218, 27, 86], [4, 238, 206], [154, 175, 116], [32, 53, 39], [67, 17, 51], [219, 100, 180], [77, 167, 173], [69, 52, 50], [122, 253, 231], [216, 131, 189], [10, 108, 28], [153, 161, 44], [118, 24, 145], [254, 162, 20], [149, 40, 39], [1, 159, 246], [206, 148, 221], [208, 107, 245], [22, 147, 126], [182, 12, 230], [109, 9, 74], [101, 142, 82], [131, 105, 106], [71, 104, 155], [186, 205, 5], [201, 76, 63], [216, 131, 143], [65, 163, 191], [147, 216, 103], [55, 113, 135], [224, 96, 0], [158, 245, 148], [40, 32, 228], [50, 34, 79], [57, 197, 191], [162, 120, 247], [181, 96, 4], [118, 236, 18], [185, 88, 97], [204, 36, 134], [166, 185, 69], [83, 12, 82], [64, 115, 17], [222, 106, 44], [248, 105, 176], [13, 248, 251], [2, 222, 160], [176, 108, 183], [216, 79, 178], [168, 90, 157], [88, 253, 107], [148, 184, 14], [195, 137, 192], [41, 254, 217], [46, 12, 46], [168, 193, 112], [106, 89, 145], [81, 239, 144], [122, 217, 206], [227, 116, 60], [81, 217, 4], [226, 53, 84], [188, 157, 106], [0, 220, 151], [41, 84, 38], [119, 175, 22], [243, 58, 213], [63, 70, 169], [66, 253, 195], [167, 105, 202], [253, 184, 227], [171, 211, 6], [166, 30, 69], [26, 43, 145], [21, 184, 153], [6, 160, 40], [13, 213, 103], [237, 248, 251], [234, 153, 211], [220, 74, 171], [169, 168, 230], [57, 156, 244], [174, 32, 113], [94, 7, 1], [180, 219, 200], [198, 159, 189], [1, 168, 110], [11, 25, 181], [148, 117, 139], [3, 242, 40], [221, 31, 228], [18, 153, 17], [184, 242, 125], [95, 213, 139], [47, 74, 98], [81, 87, 24], [156, 12, 113], [29, 62, 20], [170, 11, 100], [202, 187, 225], [249, 114, 131], [58, 225, 41], [246, 149, 225], [140, 107, 77], [70, 44, 182], [157, 35, 177], [233, 141, 129], [64, 86, 28], [134, 108, 130], [255, 150, 194], [205, 178, 25], [116, 121, 12], [215, 183, 157], [39, 87, 163], [141, 3, 141], [216, 139, 158], [76, 239, 36], [162, 75, 6], [150, 37, 233], [64, 203, 251], [211, 206, 247], [172, 127, 155], [4, 1, 18], [63, 106, 46], [201, 252, 101], [242, 45, 102], [245, 131, 108], [10, 239, 63], [46, 108, 55], [20, 3, 209], [68, 233, 223], [139, 185, 6], [153, 186, 69], [237, 145, 69], [50, 238, 44], [71, 252, 95], [202, 200, 176], [8, 139, 36], [223, 65, 67], [215, 95, 191], [124, 174, 16], [92, 185, 210], [242, 215, 43], [155, 176, 192], [160, 223, 101], [110, 165, 247], [126, 65, 60], [220, 64, 124], [205, 240, 136], [184, 229, 84], [154, 125, 36], [178, 138, 80], [126, 157, 62], [231, 192, 195], [77, 207, 76], [113, 101, 39], [233, 130, 215], [247, 211, 73], [51, 143, 80], [69, 124, 218], [124, 221, 196], [172, 185, 82], [62, 148, 240], [31, 177, 132], [192, 81, 240], [79, 66, 11], [153, 206, 200], [40, 234, 247], [197, 222, 233], [125, 42, 178], [141, 157, 106], [2, 234, 154], [88, 229, 18], [116, 14, 143], [123, 20, 21], [15, 79, 188], [87, 235, 121], [56, 234, 94], [228, 238, 183], [158, 169, 45], [151, 247, 154], [171, 69, 63], [253, 38, 174], [184, 86, 48], [225, 101, 252], [8, 42, 120], [2, 145, 168], [138, 168, 83], [93, 55, 122], [211, 95, 198], [105, 14, 198], [212, 236, 24], [69, 255, 251], [249, 221, 53], [216, 118, 152], [198, 121, 23], [7, 52, 16], [112, 102, 49], [16, 82, 90], [221, 128, 149], [22, 10, 218], [241, 47, 131], [190, 174, 133], [78, 136, 66], [52, 128, 178], [146, 36, 178], [182, 33, 43], [116, 72, 84], [7, 59, 181], [127, 130, 234], [106, 73, 16], [53, 128, 107], [38, 36, 82], [39, 41, 48], [147, 14, 178], [244, 175, 164], [154, 231, 15], [44, 0, 226], [197, 118, 147], [20, 107, 144], [57, 58, 50], [106, 152, 3], [98, 192, 192], [204, 188, 87], [222, 78, 65], [159, 159, 36], [173, 204, 94], [205, 244, 243], [250, 32, 129], [97, 250, 151], [203, 110, 151], [83, 48, 67], [112, 54, 91], [119, 105, 153], [74, 125, 163], [212, 249, 68], [99, 237, 242], [136, 197, 112], [48, 70, 56], [84, 7, 136], [131, 87, 223], [87, 42, 212], [163, 38, 93], [251, 149, 197], [210, 94, 57], [32, 248, 223], [198, 44, 108], [244, 42, 44], [131, 191, 250], [25, 169, 46], [126, 115, 118], [20, 100, 59], [64, 145, 242], [4, 223, 76], [40, 129, 168], [148, 58, 211], [34, 31, 159], [166, 39, 90], [82, 135, 27], [220, 177, 223], [88, 218, 131], [204, 233, 105], [171, 148, 166], [230, 46, 136], [227, 50, 240], [23, 223, 10], [12, 22, 61], [101, 58, 250], [126, 64, 94], [251, 93, 98], [158, 167, 85], [171, 1, 6], [216, 163, 221], [82, 50, 196], [79, 231, 32], [48, 104, 67], [229, 34, 200], [214, 239, 168], [59, 144, 170], [233, 11, 90], [67, 71, 113], [171, 204, 249], [211, 104, 70], [127, 143, 84], [3, 62, 27], [148, 237, 226], [196, 190, 146], [211, 235, 75], [242, 81, 102], [89, 213, 9], [17, 191, 176], [4, 54, 183], [145, 33, 88], [243, 123, 55], [104, 119, 235], [54, 65, 202], [147, 220, 8], [148, 75, 232], [71, 196, 228], [161, 22, 190], [68, 50, 175], [83, 207, 182], [33, 7, 42], [18, 55, 171], [150, 98, 81], [132, 243, 226], [158, 193, 222], [48, 178, 157], [83, 153, 130], [118, 48, 139], [75, 214, 174], [91, 43, 121], [118, 54, 243], [122, 245, 40], [192, 41, 185], [183, 145, 131], [149, 75, 192], [237, 68, 180], [122, 174, 167], [192, 127, 200], [192, 213, 185], [121, 234, 194], [215, 28, 134], [4, 48, 49], [151, 137, 72], [141, 190, 57], [42, 67, 77], [101, 162, 228], [175, 216, 153], [157, 173, 49], [104, 107, 195], [76, 142, 21], [167, 239, 85], [140, 234, 234], [168, 115, 233], [158, 1, 241], [108, 50, 225], [16, 187, 168], [253, 140, 117], [139, 15, 240], [243, 68, 208], [26, 9, 187], [251, 254, 92], [133, 243, 156], [74, 128, 23], [149, 79, 112], [28, 163, 126], [172, 251, 217], [212, 206, 138], [207, 167, 44], [239, 223, 58], [23, 250, 177], [163, 178, 134], [237, 154, 162], [109, 223, 7], [233, 99, 253], [245, 16, 65], [182, 148, 197], [189, 2, 70], [245, 177, 2], [89, 90, 255], [98, 32, 237], [172, 144, 138], [243, 20, 177], [63, 207, 108], [210, 66, 173], [108, 226, 176], [138, 228, 156], [97, 109, 108], [73, 206, 129], [129, 179, 66], [39, 226, 23], [159, 223, 32], [185, 85, 45], [57, 231, 143], [35, 36, 45], [212, 108, 252], [205, 0, 74], [94, 156, 127], [77, 161, 204], [247, 148, 227], [145, 212, 148], [124, 156, 127], [172, 94, 42], [240, 117, 194], [232, 176, 13], [147, 58, 117], [63, 24, 37], [141, 55, 249], [157, 171, 100], [238, 37, 163], [55, 209, 177], [205, 177, 19], [185, 61, 255], [23, 186, 220], [4, 221, 189], [0, 233, 31], [34, 30, 104], [29, 16, 145], [107, 142, 235], [196, 83, 215], [18, 164, 108], [208, 175, 141], [111, 62, 8], [30, 241, 1], [183, 97, 77], [114, 38, 127], [114, 24, 43], [0, 226, 174], [92, 111, 179], [26, 55, 16], [140, 15, 4], [146, 41, 254], [152, 135, 251], [241, 175, 44], [228, 136, 87], [6, 197, 251], [116, 161, 173], [249, 22, 240], [149, 24, 249], [28, 216, 192], [212, 170, 177], [63, 254, 54], [210, 102, 155], [195, 203, 213], [159, 4, 22], [245, 43, 110], [119, 76, 50], [46, 53, 198], [50, 96, 106], [1, 84, 77], [200, 115, 143], [149, 42, 167], [133, 125, 175], [250, 221, 139], [11, 181, 227], [225, 202, 125], [45, 204, 214], [159, 76, 207], [222, 88, 250], [12, 83, 236], [100, 164, 153], [121, 64, 21], [126, 53, 158], [119, 168, 145], [59, 180, 193], [46, 31, 21], [125, 231, 76], [169, 220, 77], [110, 4, 232], [157, 122, 74], [173, 207, 27], [124, 29, 113], [107, 10, 125], [208, 142, 119], [122, 50, 178], [155, 55, 1], [167, 203, 55], [196, 33, 112], [76, 69, 117], [153, 251, 252], [128, 138, 20], [156, 251, 138], [63, 147, 162], [204, 42, 166], [213, 41, 106], [251, 94, 34], [1, 90, 60], [50, 132, 48], [148, 125, 177], [158, 18, 81], [35, 3, 54], [93, 22, 172], [140, 95, 205], [128, 176, 21], [156, 74, 8], [142, 206, 216], [215, 147, 159], [1, 82, 142], [123, 26, 119], [195, 229, 246], [49, 7, 205], [230, 206, 44], [185, 92, 197], [35, 80, 146], [167, 16, 174], [152, 138, 89], [177, 11, 53], [193, 209, 180], [165, 179, 105], [34, 64, 173], [177, 41, 146], [143, 201, 104], [70, 35, 114], [99, 18, 162], [242, 92, 117], [43, 169, 2], [143, 183, 198], [200, 111, 18], [231, 68, 12], [195, 18, 228], [230, 43, 188], [146, 72, 157], [127, 66, 232], [21, 116, 182], [252, 247, 248], [178, 142, 83], [224, 225, 119], [78, 5, 23], [17, 82, 65], [160, 70, 111], [31, 24, 74], [36, 117, 23], [166, 82, 1], [201, 175, 45], [75, 80, 180], [241, 222, 200], [103, 136, 58], [194, 151, 19], [13, 73, 150], [253, 217, 151], [73, 33, 58], [171, 177, 59], [101, 20, 41], [6, 34, 254], [126, 131, 211], [203, 170, 209], [131, 241, 132], [162, 196, 245], [248, 230, 155], [247, 66, 6], [179, 74, 238], [246, 10, 138], [75, 195, 148], [42, 183, 13], [147, 70, 143], [217, 113, 253], [76, 230, 170], [57, 214, 166], [180, 94, 176], [235, 104, 41], [70, 33, 66], [130, 36, 175], [254, 229, 113], [24, 25, 127], [95, 104, 157], [173, 127, 207], [140, 8, 125], [175, 142, 58], [140, 65, 177], [27, 255, 51], [26, 22, 114], [12, 193, 238], [18, 246, 241], [115, 219, 1], [234, 89, 232], [239, 131, 226], [212, 42, 110], [177, 145, 93], [240, 54, 49], [240, 105, 2], [204, 136, 178], [227, 197, 229], [88, 16, 138], [65, 214, 168], [219, 242, 76], [228, 89, 92], [243, 246, 114], [31, 138, 190], [63, 201, 26], [159, 105, 186], [50, 213, 117], [213, 222, 222], [164, 44, 55], [207, 91, 78], [140, 220, 90], [49, 124, 181], [189, 255, 109], [136, 231, 217], [170, 128, 11], [54, 38, 169], [117, 160, 154], [123, 26, 11], [233, 165, 194], [143, 236, 100], [193, 249, 21], [148, 215, 196], [228, 163, 133], [111, 254, 247], [142, 81, 132], [158, 72, 35], [237, 144, 252], [231, 87, 237], [51, 50, 207], [83, 125, 87], [50, 14, 134], [37, 214, 139], [235, 254, 21], [6, 62, 1], [150, 253, 197], [124, 217, 185], [202, 79, 106], [9, 171, 85], [60, 250, 121], [157, 253, 248], [206, 9, 100], [232, 93, 35], [116, 3, 16], [219, 212, 104], [255, 144, 14], [171, 61, 139], [26, 210, 187], [25, 247, 214], [54, 175, 31], [162, 182, 81], [116, 150, 231], [71, 122, 125], [154, 38, 229], [128, 216, 166], [119, 170, 91], [0, 9, 247], [206, 106, 47], [114, 144, 185]]

    def __init__(self, **kwargs):
        super(NyuDataset, self).__init__(
            **kwargs
        )
