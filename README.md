https://github.com/user-attachments/assets/2973c8ed-196f-47da-8b6e-757a26dd4f65

# Accessible Vehicle Detection

🏆 **1st Place – ParkHub Track, Southern Methodist University Hackathon**

We developed a **real-time computer vision algorithm** to detect accessible vehicles in video streams by identifying the **International Symbol of Access** (ISA) on license plates, windows, and other areas of the vehicle.

---

## Problem Statement

Create an algorithm that:

- Detects the ISA without identifying or inferring information about people.
- Uses only open datasets.
- Processes 30 fps video in real time (**< 33 ms per frame**) without frame drops.
- Outputs bounding boxes and confidence scores for detected accessible vehicles.

---

## Judging Criteria

- **Speed:** Under 33 ms per frame (faster is better)  
- **Memory Efficiency:** Lower is better  
- **Accuracy:** Correctly detect accessible vehicles in arbitrary video streams  

---

## Tech Stack

- **Language:** Python  
- **Libraries:** OpenCV, TensorFlow / PyTorch (depending on final model)  
- **Dataset:** Publicly available open datasets with ISA symbols  

---

## How It Works

1. Preprocess video frames for optimal detection speed and accuracy.  
2. Use a trained deep learning model to identify ISA symbols in each frame.  
3. Draw bounding boxes around detected symbols and display confidence scores.  
4. Output results in real time without dropping frames.

---

## Contributors

- *Abdul Mannan* – Model Development, Integration  
- *Usama Bin Faheem* – Data Preparation, Optimization, Testing




![1728370147204](https://github.com/user-attachments/assets/21413702-a8c7-4261-9e87-5498f1afad61)
