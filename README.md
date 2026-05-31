# AI Spatial Optimization with Gemini & LlamaIndex 🗺️🤖

## Overview
This repository contains two AI-driven Python scripts developed during Artificial Intelligence Hackathons. Both projects solve complex spatial optimization problems on grid-based maps by combining classic algorithmic techniques with the reasoning capabilities of **Google's Gemini 2.5 Flash** model via **LlamaIndex Workflows**.

The core objective of these scripts is to strategically place infrastructure (electrical transformers or waste containers) to minimize global distances to target nodes (hospitals, industries, or houses) while adhering to strict placement rules.

## Projects Included

### 1. Power Grid Optimizer (`MarGav.py`) 
Optimizes the placement of 6 electrical transformers ('C') on a city grid to supply Industries ('T') while minimizing the distance to Hospitals ('O').
* **Strategic Filtering:** Implements a custom heuristic function to pre-select candidate coordinates (must touch a road 'X', cannot touch restricted buildings 'E', prioritizing proximity to 'T' and 'O').
* **AI Evaluation Loop:** Uses LlamaIndex `Workflow` to pass filtered candidates to Gemini. The LLM reasons the best combination, which is then strictly parsed, validated, and scored.
* **Iterative Improvement:** Feeds the evaluation score back to the LLM in a multi-attempt loop to break previous records and find the absolute minimum distance.

### 2. Urban Infrastructure Placer (`S^2.py`) 
Places 6 containers ('C') to minimize the total walking distance from all residential houses ('O') on the map.
* **Density-Based Generation:** Calculates local housing density using a specific radius to score potential coordinates.
* **Non-Maximum Suppression (Dispersion):** Ensures selected candidates are geographically distributed across the map to avoid clustering in a single high-density zone.
* **BFS Distance Calculation:** Uses Breadth-First Search (BFS) to accurately calculate the Manhattan distance from houses to the nearest container, even avoiding obstacles.

## Technologies & Architecture
* **Python 3**
* **Google Gemini API** (`gemini-2.5-flash` for high precision and speed).
* **LlamaIndex:** Utilizes Event-Driven Workflows (`StartEvent`, `GenerationEvent`, `EvaluationEvent`) to create an autonomous agent loop.
* **Algorithmic Design:** Breadth-First Search (BFS), Non-Maximum Suppression, JSON parsing with RegEx fallbacks, and Grid/Matrix manipulation.

## Security Note
*For security and compliance reasons, the Google GenAI API keys have been removed from the source code (`"Insert your API key"`). To run these scripts locally, please provide your own valid API key in the configuration section.*
