# Pushing SAM to Its Limits: A Journey Through Three Specialized Computer Vision Challenges

*Final project for Computer Vision class by Anurag Dhungana and Prakriti Bista*

## Introduction

**Can a generalist model learn to become a specialist?** That’s the question we set out to answer when we decided to fine-tune the Segment Anything Model (SAM) for three wildly different segmentation tasks. What started as an exploration of SAM’s adaptability turned into a fascinating lesson about when specialization works, when it fails spectacularly, and what factors make the difference.

This is the story of our journey through wood texture discrimination, industrial defect detection, and camouflaged object segmentation, complete with the failures, the breakthroughs, and everything we learned along the way and a *visit* to the Smithsonian National Zoo.

## The Big Idea: Can We Teach SAM New Tricks?

SAM is incredible at what it does. Give it any image, point anywhere, and it’ll segment something. But “good at everything” often means “great at nothing specific.” We wanted to know: could we take SAM’s powerful foundation and specialize it for tasks where the vanilla model struggles?

Our approach is simple:
1. **Freeze the massive image encoder** 
2. **Pre-compute image embeddings** once and save them 

This gave us the ability to iterate quickly and test our hypotheses across three very different domains. We started with what seemed like the easiest task and worked our way up to the hardest. 

## Why These Three Challenges? Real-World Motivation

Before diving into the technical work, it’s worth explaining why we chose these specific domains. Each represented what we felt was a different real-world application and a different level of difficulty.

### Wood Texture Segmentation: The Starting Point

This was just the first dataset we thought of. We wanted to start with something that seemed straightforward, teaching SAM to distinguish between different wood grains. It felt like a good warm-up exercise, a chance to get our fine-tuning pipeline working before tackling harder problems.

In hindsight, we probably should have asked “Does this actually need solving?” before jumping in. But sometimes you learn the most from the mistakes you make at the beginning.

### Carpet Defect Detection: The Industrial Vision

**Real-world scenario:** Imagine a carpet manufacturing facility where thousands of meters of fabric roll off production lines every day. Currently, quality control relies on human inspectors visually scanning for defects, cuts, thread pulls, color inconsistencies, and foreign material contamination. It’s tedious, it’s expensive, and humans naturally miss subtle defects after hours of inspection.

Our vision was to create an automated quality control system:
- Cameras mounted above the production line capture continuous images
- Our specialized SAM processes each frame in real-time
- When a defect is detected, the system alerts operators and marks the location
- Reduces waste by catching defects before further processing
- Improves consistency and frees human inspectors for more complex judgments

The challenge is that carpet defects are often subtle, a barely visible cut, a slight color variation, a single broken thread in a complex weave pattern. Base SAM treats the entire carpet surface as one uniform object. We needed to teach it that “texture interruption = defect boundary.”

This is a real problem in textile manufacturing, and a solution could save companies significant money while improving product quality.

### Camouflaged Object Detection: The Wildlife Application

**Real-world scenario:** You’re on a safari, or visiting a zoo, or hiking through a rainforest. There’s incredible wildlife all around you, but you can’t see it. That lizard on the tree trunk? Invisible. The moth on the bark? You’d never spot it. The stick insect is literally three feet away? It might as well not exist.

Now imagine having a tool that could help:
- **Safari guides** could use it to quickly spot camouflaged animals for tourists
- **Wildlife photographers** could identify hidden subjects that would make stunning shots
- **Biologists** conducting field research could detect species that evade visual surveys
- **Zoo educators** could point out camouflaged animals in exhibits to help visitors appreciate them
- **Conservation efforts** could use it to count populations of species that are difficult to observe

Beyond recreation and education, this has serious scientific applications. Many ecological surveys rely on visual detection, but camouflaged species are systematically undercounted. A tool that can reliably detect camouflaged animals could improve biodiversity assessments and conservation planning.

The technical challenge is fascinating: these animals have evolved for millions of years specifically to defeat visual detection systems (predators, prey, humans). Can we teach an AI to see what evolution designed to be invisible?

## Challenge 1: Wood Texture Segmentation - When “Too Good” Becomes a Problem

### Wood Texture Dataset Structure

```
wood_data/
├── images/
│   └── images/
│       ├── abura-s-60x60.jpg
│       ├── african-blackwood-60x60.jpg
│       ├── oak-60x60.jpg
│       ├── walnut-60x60.jpg
│       └── ... (530 total images)
└── wood.csv (metadata with species names)
```

Each image is a 60×60 or similar resolution texture sample. We created 500 synthetic 512×512 mosaics (2×2 grids of 256×256 tiles) for training.

### The Setup

**Dataset:** 530 wood texture images from various species (oak, walnut, pine, etc.)
**Model:** MobileSAM (the lightweight variant with ~40M parameters)
**Training Data:** 500 synthetic 2×2 mosaics (each containing 4 different wood textures)
**The Goal:** Teach SAM to distinguish between visually similar wood grains

We started by creating “mosaic” images, think of them as 2×2 grids where each quadrant contains a different wood texture. To make things interesting, we used ResNet18 to extract features from all 530 images, then applied K-Means clustering to group similar woods together. 

This gave us 5 distinct clusters:

```
Cluster 1: 183 images (34.5%) - Lighter hardwoods
Cluster 0: 171 images (32.3%) - Medium tones
Cluster 2:  96 images (18.1%) - Darker woods
Cluster 4:  41 images (7.7%)  - Exotic species
Cluster 3:  39 images (7.4%)  - Very distinct grains
```

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image.png)

*Figure:  image showing the wood texture dataset structure and sample mosaics*

### Baseline Evaluation (Zero-Shot Mobile Sam)

We gave MobileSAM the *perfect center point* of each tile. The thought here was: 

- If it's a good generalist, it should segment the square tile perfectly.
- If it fails, it will likely over-segment (texture details) or under-segment (merge similar woods).

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%201.png)

*Figure: Zero shot (Average Baseline IoU for this image: 0.9950)*

### The “Hard Mode” Challenge

Here’s where it got interesting. We created two test modes:
- **Standard Mode:** Mosaics with textures from different clusters ( lots of visual contrast)
- **Hard Mode:** Mosaics where ALL four tiles came from the SAME cluster ( similar colors, only grain direction differs)

The hypothesis was simple: if our specialized model could learn fine-grained texture boundaries, it should outperform vanilla SAM on Hard Mode.

### The Training

We ran a straightforward training loop:
- **5 epochs**
- **Learning rate:** 1×10⁻⁴
- **Loss function:** Dice loss only
- **Prompt strategy:** Perfect center point of each tile

The training seemed to go great. The loss dropped beautifully:

```
Epoch 1: Loss = 0.1570
Epoch 2: Loss = 0.0125  (92% drop!)
Epoch 3: Loss = 0.0426  (wait, it went UP?)
Epoch 4: Loss = 0.0002  (okay, back down...)
Epoch 5: Loss = 0.0001  (basically zero!)
```

We were thrilled. Loss of 0.0001? That’s incredible! Time to test it.

### The Devastating Results

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%202.png)

|  | **Base MobileSAM** | **Our “Specialized” SAM** | **Change** |
| --- | --- | --- | --- |
| **Hard Mode mIoU** | **95.80%** | 59.52% | **-36.28%**  |

*Figure: side-by-side comparison images showing base SAM vs specialized SAM on Hard Mode*

We didn’t just fail to improve, we made it **36% WORSE**.

On standard mosaics, vanilla MobileSAM was already crushing it at 99.50% IoU. But on Hard Mode (similar textures), base SAM scored 95.8% while our “specialized” version completely collapsed to 59.5%.

### What Went Horribly Wrong

Looking back at that training loss, the problem was obvious: **the model had memorized the training set**. Loss of 0.0001 isn’t learning - it’s overfitting.

Here’s what killed us:

1. **Insufficient Data:** 500 training mosaics = ~2,000 individual tile instances. 
2. **Dice Loss Alone:** We only used Dice loss, which measures region overlap. Without pixel-level supervision (Binary Cross-Entropy), the model wasn’t penalized for noisy boundaries. It learned to draw “blobby” masks that worked on training data but failed on anything new.
3. **No Validation Set:** We had no way to detect overfitting until it was too late. The model happily kept training long after it stopped generalizing.
4. **Base SAM Was Already Too Good:** When the baseline is 95.8%, there’s not much room for improvement. We tried to fix something that wasn’t broken and ended up breaking it.
5. **The Instability Spike:** That jump from 0.0125 to 0.0426 at epoch 3? Red flag. The learning rate was too high and the optimization was unstable.

### What We Learned

This failure taught us more than a success would have:
- **Don’t fix what ain’t broke:** If base SAM scores >95%, specialization is risky
- **Validation is non-negotiable:** Always monitor generalization
- **Low loss ≠ good model:** Loss near zero is usually a warning sign
- **Data quantity matters:** 500 samples isn’t enough for subtle distinctions
- **Loss function composition matters:** Single-component losses are dangerous

## Challenge 2: Carpet Defect Detection - The Synthetic Data Experiment

### The Setup

**Dataset:** MVTec AD Carpet Dataset
**Model:** MobileSAM (same as wood experiment)
**Real Training Data:** 280 defect-free “good” carpet images
**Real Test Data:** 89 images with real industrial defects (5 types: cuts, holes, thread pulls, metal contamination, color defects)
**The Problem:** 89 defect images is way too few to train a neural network

This was an industrial anomaly detection challenge. Imagine a carpet factory where cameras scan every meter of output looking for manufacturing flaws. These defects are subtle, random, and expensive to collect labeled examples of.

### Carpet Defect Dataset Structure

```
carpet/
├── train/
│   └── good/ (280 defect-free images)
├── test/
│   ├── color/ (19 images)
│   ├── cut/ (15 images)
│   ├── hole/ (16 images)
│   ├── metal_contamination/ (20 images)
│   └── thread/ (19 images)
└── ground_truth/
    ├── color/ (*_mask.png files)
    ├── cut/
    ├── hole/
    ├── metal_contamination/
    └── thread/
```

Total: 280 training images (all good), 89 test images with defects. We generated 500 synthetic defect images from the good images for training.

### The Synthetic Data Solution

Since we couldn’t train on real defects (too few), we built a **synthetic defect generator** that could inject realistic-looking anomalies into the 280 good images:

**Type 1: Patches (simulates holes, metal contamination, cuts)**
- Cut out random 20-60px patches from the carpet
- Paste them elsewhere, but darkened (40-70% brightness) or with added Gaussian noise
- Creates the appearance of holes or foreign objects

**Type 2: Scars (simulates thread pulls, cuts)**
- Draw thin random lines (2-5px thick) across the carpet
- Use dark colors (0-50 RGB values) to simulate broken threads
- Random start/end points to create irregular patterns

**Type 3: Stains (simulates color defects)**
- Apply circular color shifts (40-100px radius) using HSV manipulation
- Shift the hue by 20-50 degrees
- Blend at 40% opacity for a realistic “dye bleeding” effect

We generated **500 synthetic training samples** this way, each one with a known mask and a center-point prompt at the anomaly location.

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%203.png)

*Figure: Image showing the synthetic defect generation pipeline with examples of each type*

### The Training

Learning from the wood disaster, we kept a close eye on the training dynamics:

```
Epoch 1: Loss = 0.2544
Epoch 2: Loss = 0.2006  (21% drop - healthy)
Epoch 3: Loss = 0.1818  (9% drop)
Epoch 4: Loss = 0.1639  (10% drop)
Epoch 5: Loss = 0.1527  (7% drop)
```

Much better! No collapse, no spikes. The loss was higher than the wood experiment (0.15 vs 0.0001), but that’s actually a good sign - it means the model was still generalizing, not memorizing.

**Training config:**
- 5 epochs
- Learning rate: 1×10⁻⁴
- Loss: Dice only (we hadn’t learned that lesson yet)
- 500 synthetic samples, tested on 89 REAL defects

### The Results: Modest but Real Improvement

|  | **Base MobileSAM** | **Specialized SAM** | **Change** |
| --- | --- | --- | --- |
| **mIoU (Real Defects)** | 29.43% | 36.18% | **+6.75%**  |

Not spectacular, but positive! We improved by 6.75 percentage points (a 23% relative improvement) without the model ever seeing a single real defect during training.

 

These are some outputs from the Base Model

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%204.png)

*Figure: Visual comparison showing Baseline SAM*

These are some of the outputs from our specialized model:

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%205.png)

*Fig: Visual comparison showing Specialized SAM(Our model)*

### Why It Worked (Sort Of)

The key insight: **our synthetic defects captured the concept of “texture interruption”** even if they didn’t perfectly match real manufacturing flaws.

Base SAM at 29% mIoU was really struggling. It treated the entire carpet as one homogeneous region. Our specialized model learned that sharp changes in texture pattern = object boundary, which helped it detect real defects even though they looked different from the synthetic training data.

But 36% mIoU still leaves 64% room for improvement. The **sim-to-real gap** was the limiting factor:
- Our synthetic “scars” were just straight lines; real thread pulls have shadows, fraying, and 3D structure
- Our “patches” were simple cut-and-paste; real metal contamination has reflective properties
- Our “stains” were smooth color shifts; real dye defects have irregular bleeding patterns

### What We Learned

- **Synthetic data can work** when real data is scarce
- **Stable training curves** (gradual loss decrease) predict better generalization than rapid collapse
- **Start from a struggling baseline:** 29% → 36% is progress; 95% → 59% is disaster(wood)
- **Domain gap is real:** Synthetic training + real testing = limited but useful improvement
- **We still needed better loss functions** (we were still using Dice-only)

---

## Challenge 3: Camouflaged Animal Detection

### The Setup

**Dataset:** COD10K (Camouflaged Object Detection, 10,000 images)
**Model:** SAM ViT-H (the BIG one—636M parameters, 2.4GB checkpoint)

GPU Used
**Training Data:** 6,000 images of camouflaged animals (augmented to 12,000 with horizontal flips)
**Test Data:** 4,000 images (we evaluated on 200 random samples)
**The Goal:** Detect animals that evolution has designed to be invisible

### Camouflage Dataset (COD10K-v3)

```
COD10K-v3/
├── Train/
│   ├── Image/ (6,000 images)
│   │   ├── COD10K-CAM-1-Aquatic-1-BatFish-1.jpg
│   │   ├── COD10K-CAM-1-Aquatic-2-Crab-10.jpg
│   │   └── ...
│   └── GT_Object/ (6,000 binary masks)
└── Test/
    ├── Image/ (4,000 images)
    └── GT_Object/ (4,000 masks)
```

After horizontal flip augmentation: **12,000 training samples**. The test set contained 4,000 images; we evaluated on 200 randomly selected samples for efficiency.

### The Dataset

COD10K contains images organized by categories:
- Aquatic animals (fish, octopuses, seahorses camouflaged against reefs)
- Terrestrial animals (insects, lizards, frogs blending into leaves/bark)
- Flying animals (moths with wing patterns matching tree bark)

Each image has a corresponding binary mask showing exactly where the camouflaged animal is. The dataset is challenging because:
- **Low contrast:** Animal and background have similar colors
- **Complex textures:** Natural environments have high-frequency patterns everywhere
- **Scale variation:** Animals range from tiny insects to large reptiles
- **Occlusion:** Many animals are partially hidden

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%206.png)

*Figure: Showing sample: COD10K-CAM-1-Aquatic-1-BatFish-1.jpg*

### Learning From Our Mistakes

This time, we applied everything we’d learned from the previous failures:

**Change 1: Way More Data**
- Wood: 500 samples → FAIL
- Carpet: 500 samples → Modest success
- **Camouflage: 6,000 samples → 12,000 with augmentation** 

**Change 2: Bigger Model**
- Wood/Carpet: MobileSAM (40M parameters)
- **Camouflage: SAM ViT-H (636M parameters)** 
- The huge vision transformer has a much richer semantic understanding

**Change 3: Combined Loss Function**

```python
bce_loss = Binary_Cross_Entropy(pred, target)  # Pixel-level accuracydice_loss = 1 - Dice_Coefficient(pred, target)  # Region overlapfinal_loss = 0.5 * bce_loss + 0.5 * dice_loss
```

No more Dice-only! BCE forces pixel-perfect boundaries while Dice maintains shape coherence.

**Change 4: Multi-Prompt Training**
Instead of just using point prompts, we randomly switched between:
- **Point prompts** (50%): Single click at foreground object center
- **Bounding box prompts** (50%): Tight box around the animal

This made the model robust to different prompt types and improved generalization.

### The Training

Seven epochs of stable, beautiful learning:

```
Epoch 1: Loss = 0.1420
Epoch 2: Loss = 0.1272  (10.4% drop)
Epoch 3: Loss = 0.1172  (7.9% drop)
Epoch 4: Loss = 0.1140  (2.7% drop)
Epoch 5: Loss = 0.1113  (2.4% drop)
Epoch 6: Loss = 0.1081  (2.9% drop)
Epoch 7: Loss = 0.1059  (1.8% drop)

Total: 25.4% reduction, NO COLLAPSE
```

We had a gradual, consistent improvement. Final loss of 0.1059 (not 0.0001, like it was before). This is what we felt was healthy training looks like.

### The Breakthrough Results

|  | **Base SAM ViT-H** | **Specialized SAM** | **Change** |
| --- | --- | --- | --- |
| **mIoU (200 test samples)** | 47.52% | **66.35%** | **+18.83%**  |

**We did it.** An 18.83% point improvement, a 39.6% relative gain on one of the hardest segmentation tasks in computer vision. Here are some of the results. There were some where both of them were able to have the same performance, but a lot of them ours was much better. (Double clicking on the images expands it)

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%207.png)

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%208.png)

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%209.png)

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%2010.png)

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%2011.png)

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%2012.png)

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%2013.png)

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%2014.png)

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%2015.png)

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%2016.png)

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%2017.png)

*Figure: Comprehensive visual comparison showing 8-10 examples with Input/Base SAM/Specialized SAM side-by-side*

### Why It Worked

**1. Base SAM Was Actually Struggling**
At 47.52% mIoU, vanilla SAM was barely better than random. Why? Because SAM relies heavily on edge contrast. When a green lizard sits on a green leaf with the same color, SAM sees one object (the leaf). There’s genuine room for improvement here, unlike the wood experiment, where base SAM was already at 95%.

**2. We Had Enough Data**
12,000 training samples gave the model enough examples to learn the pattern: *“texture disruption matters more than color matching.”* A stick insect and a twig might be the same brown, but the insect’s legs create subtle discontinuities in the bark pattern, and our model learned to detect these.

**3. The Loss Function Did Its Job**
- **Dice loss** kept the overall shape coherent 
- **BCE loss** forced precise boundaries 
- Together, they created tight, accurate masks

**4. Multi-Prompt Training Made It Robust**
By training on both points and boxes, the model couldn’t rely on any one input pattern. It had to actually understand where the camouflaged object was, not just memorize prompt types.

**5. The Big Model Had Capacity**
ViT-H’s 636M parameters gave it the representational power to learn subtle texture disruption patterns that MobileSAM’s 40M parameters couldn’t capture.

### Real-World Test: The Smithsonian Zoo

We didn’t just test on the COD10K dataset; we took our model to the **Smithsonian National Zoo** and photographed real animals in their enclosures to see how it performed on completely new data.

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%2018.png)

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%2019.png)

*Figure: Images from our visit to the Zoo*

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%2020.png)

![image.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/image%2021.png)

*Figure: Images from Google.*

## Multi-point comparison

![comprehensive_evaluation_comparison.png](Pushing%20SAM%20to%20Its%20Limits%20A%20Journey%20Through%20Three%20/comprehensive_evaluation_comparison.png)

| model | prompt_strategy | iou_mean | iou_std | dice_mean | dice_std | f1_mean | f1_std | boundary_prec_mean | boundary_prec_std | boundary_recall_mean | boundary_f1_mean | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base SAM ViT-H | Center-of-Mass (Single) | 0.47522295555813043 | 0.39340137641139583 | 0.5346420742957994 | 0.4141000472614373 | 0.5346420742957995 | 0.4141000472614373 | 0.5671298172864384 | 0.3437546778738809 | 0.5106681594464915 | 0.4976784545300676 | 200 |
| Base SAM ViT-H | Edge (Single) | 0.22723412304536045 | 0.3224685560334072 | 0.27619974680708764 | 0.360949503438628 | 0.27619974680708764 | 0.360949503438628 | 0.4590291973792144 | 0.3380149045749537 | 0.3111292644042726 | 0.3025289410123832 | 200 |
| Base SAM ViT-H | Multi-Point Grid (4 pts) | 0.6080339518005529 | 0.3519338556086735 | 0.6780094551782202 | 0.3609438366587571 | 0.6780094551782202 | 0.36094383665875707 | 0.6615792103693227 | 0.3155453646613194 | 0.5931974370426463 | 0.5929179360348111 | 200 |
| Specialized SAM ViT-H | Center-of-Mass (Single) | 0.6572620786930107 | 0.3029298361216438 | 0.7397729298017715 | 0.2961040228777779 | 0.7397729298017715 | 0.2961040228777779 | 0.6720444778176347 | 0.30051660410446285 | 0.6784726060433367 | 0.6527537761728501 | 200 |
| Specialized SAM ViT-H | Edge (Single) | 0.6463239918083087 | 0.2977494671406225 | 0.734018792668306 | 0.28710417931371995 | 0.734018792668306 | 0.2871041793137199 | 0.6323785695590518 | 0.3053824886822996 | 0.6949795103354347 | 0.6375097254920465 | 200 |
| Specialized SAM ViT-H | Multi-Point Grid (4 pts) | 0.6680696881087215 | 0.28977162696875114 | 0.7530656712001675 | 0.27997131961340327 | 0.7530656712001673 | 0.27997131961340327 | 0.6801212192159919 | 0.29854258829013314 | 0.6801048608696503 | 0.6573306856627894 | 200 |

## The Comparative Analysis

Let’s put all three experiments side by side and extract the universal lessons.

### Performance Summary

| **Experiment** | **Model** | **Training Samples** | **Base mIoU** | **Specialized mIoU** | **Change** | **Outcome** |
| --- | --- | --- | --- | --- | --- | --- |
| Wood (Hard Mode) | MobileSAM | 500 mosaics (~2K tiles) | 95.80% | 59.52% | **-36.28%** |  Failed |
| Carpet Defects | MobileSAM | 500 synthetic | 29.43% | 36.18% | **+6.75%** |  Modest |
| Camouflage | SAM ViT-H | 12,000 augmented | 47.52% | 66.35% | **+18.83%** |  Success |

## Conclusion: When Specialization Works

After three experiments, one spectacular failure, one modest success, and one breakthrough, we can finally answer our original question:

**Yes, you can teach SAM new tricks, but only under the right conditions.**

Specialization works when:

1. **Base SAM is struggling** (mIoU < 75%): there's room to improve.
2. **You have sufficient data** (5,000+ samples): enough to learn patterns without overfitting.
3. **You use proper loss functions** (BCE + Dice): combining pixel-level and region-level supervision.
4. **You have model capacity** (ViT-H for complex tasks): enough parameters to capture subtle patterns.
5. **You monitor generalization** (validation sets, stable training curves): catching overfitting early.

Our camouflage model proved that well-executed specialization can deliver an **18.83% absolute improvement** on one of computer vision's hardest segmentation tasks. That's the difference between "barely better than random" (47%) and "actually useful" (66%).

Our wood texture failure was equally instructive. Trying to improve an already excellent baseline (95.8%) with insufficient data (500 samples) and a naive loss function (Dice only) led to catastrophic degradation (−36%).

The lesson? **Don't specialize for the sake of specializing.** Only do it when you have a clear problem, the right tools, and enough data to do it right.

### Factor Analysis: What Predicted Success?

**Factor 1: Room for Improvement**

The single best predictor of whether specialization works is: *How much is the baseline struggling?*

- Wood: Base at 95.8% → Little room to improve, high risk of degradation.
- Carpet: Base at 29.4% → Significant room, modest gains possible.
- Camouflage: Base at 47.5% → Perfect sweet spot, struggling but not hopeless.

**Rule of thumb:** Only specialize SAM when baseline mIoU < 75%

**Factor 2: Training Data Volume**

| Experiment | Effective Training Samples | Result |
| --- | --- | --- |
| Wood | ~2,000 tile instances | Overfitting |
| Carpet | 500 synthetic images | Limited generalization |
| Camouflage | **12,000 real images** | Strong generalization |

**Rule of thumb:** You need at least 5,000+ diverse samples for fine-grained segmentation tasks. More data smooths the loss landscape and prevents memorization.

**Factor 3: Loss Function Composition**

| Experiment | Loss Function | Final Training Loss | Test Performance |
| --- | --- | --- | --- |
| Wood | Dice only | 0.0001 (collapsed) | Catastrophic failure |
| Carpet | Dice only | 0.1527 (stable) | Modest improvement |
| Camouflage | **50% BCE + 50% Dice** | 0.1059 (stable) | Major improvement |

The combined loss was the secret. Here's why both components matter:

```python
# Dice Loss: Measures region overlap (shape-focused)dice = 1 - (2 * intersection) / (pred_area + gt_area)
# Good: Maintains overall shape coherence# Bad: Doesn't care about exact boundaries# Binary Cross-Entropy: Measures pixel-by-pixel accuracybce = -mean(gt * log(pred) + (1-gt) * log(1-pred))
# Good: Forces precise boundaries# Bad: Can over-penalize small shape errors# Combined: Best of both worldsloss = 0.5 * bce + 0.5 * dice
```

**Rule of thumb:** Never use single-component loss functions for segmentation. Always combine region-level and pixel-level losses.

**Factor 4: Model Capacity**

| Model | Parameters | Image Encoder | Used For | Result |
| --- | --- | --- | --- | --- |
| MobileSAM | ~40M | Tiny-ViT (distilled) | Wood, Carpet | 1 failure, 1 modest success |
| SAM ViT-H | ~636M | Vision Transformer Huge | Camouflage | Major success |

The 16× difference in model size mattered. The huge transformer could learn subtle texture disruption patterns that the tiny model couldn't represent.

**Rule of thumb:** For semantically complex tasks (like understanding camouflage), use the biggest model you can afford. For simple tasks (like detecting color changes), smaller models work fine.

---

**Factor 5: Training Stability**

The training curves tell you everything:

**Wood (Unstable):**

```
Epoch 1: 0.1570 → Epoch 2: 0.0125 (92% drop!)
→ Epoch 3: 0.0426 (spike!) → Epoch 5: 0.0001 (collapse)
```

Huge swings = overfitting + optimization instability

**Carpet (Stable but Limited):**

```
Epoch 1: 0.2544 → Gradual decrease → Epoch 5: 0.1527
```

Smooth curve = healthy training, but limited data caps the ceiling

**Camouflage (Ideal):**

```
Epoch 1: 0.1420 → Consistent decrease → Epoch 7: 0.1059
```

Perfect balance, learning without overfitting

**Rule of thumb:** If your training loss drops below 0.01, you're probably overfitting. Below 0.001, you're definitely overfitting. At 0.0001, you've memorized the training set.

## What We'd Do Differently Next Time

### For Wood Textures (If We Tried Again)

1. **Accept the baseline:** Base SAM at 95.8% is already excellent. This task likely doesn't need specialization.
2. **If we insisted on trying:**
    - Generate 5,000+ mosaics instead of 500
    - Use a validation set (20% split)
    - Start with learning rate 1×10⁻⁵ instead of 1×10⁻⁴
    - Use combined loss (BCE + Dice)
    - Add cosine annealing learning rate schedule
    - Stop when validation IoU stops improving
3. **Better evaluation:** Test across a wider range of difficulties, not just "hard mode."

### For Carpet Defects

1. **Generate 2,000+ synthetic defects** instead of 500
2. **Improve synthetic realism:**
    - Add shadows to scars and patches
    - Use more sophisticated cut-paste with boundary smoothing
    - Generate composite defects (scar + stain simultaneously)
    - Add Gaussian blur to simulate depth-of-field
3. **Use SAM ViT-H instead of MobileSAM** (bigger model for better features)
4. **Combined loss function** (BCE + Dice)
5. **More aggressive augmentation:**
    - Random rotations (0–360°)
    - Random scaling (0.8–1.2x)
    - Color jitter (brightness, contrast, saturation)
    - Elastic deformations to simulate carpet warping

**Expected outcome with these changes:** 29% → 50 - 55% mIoU (vs. the 36% we achieved)

### For Camouflage (What We Got Right)

We nailed this one. The only improvements would be:

1. **Learning rate scheduling:** Cosine annealing would let us train longer without overfitting
2. **More augmentations:**
    - Random brightness/contrast (simulate different lighting)
    - Random zoom crops (teach scale invariance)
3. **Per-category analysis:** Evaluate separately on aquatic, terrestrial, and flying animals to find weak spots
4. **Boundary-focused loss:** Add a boundary loss term to specifically penalize edge errors

**Expected outcome:** 66% → 72 - 75% mIoU

---

## Resources & Code

**GitHub Repository:** [https://github.com/Aarekaz/Special-SAM](https://github.com/Aarekaz/Special-SAM)

**Pretrained Models:**
- Specialized Camouflage SAM: [https://huggingface.co/AAREKAZ/SpecialSAM/tree/main](https://huggingface.co/AAREKAZ/SpecialSAM/tree/main)

**Datasets Used:**
- MVTec AD Carpet: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- COD10K: [https://www.kaggle.com/datasets/aarekaz/cod10k](https://www.kaggle.com/datasets/aarekaz/cod10k)