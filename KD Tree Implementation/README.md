# EKD Tree Project

## Overview
This project implements an EKD Tree (Enhanced KD-tree), a data structure used for organizing multi-dimensional data. It supports efficient spatial operations such as nearest neighbor search and range queries.

## Features
- Insert points into the tree  
- Delete points from the tree  
- k-Nearest Neighbor (k-NN) search  
- Range queries (find all points within a bounding box)  
- Tree visualization via JSON dump  

## File Structure
- `ekd.py`  
  Contains the full EKD Tree implementation, including all core operations.

- `test_ekd.py`  
  Reads commands from a trace file and executes operations on the tree.

## How It Works

### Tree Structure
- Each node is either:
  - **Leaf node**: stores a list of points  
  - **Internal node**: splits data along a dimension  

- Splitting is based on:
  - **cycle method** (round-robin over dimensions), or  
  - **spread method** (dimension with largest spread)

### Supported Operations
- **insert(point, code)**  
  Adds a new point to the tree.

- **delete(point)**  
  Removes a point from the tree.

- **knnquery(k, point)**  
  Returns the k nearest neighbors to a given point.

- **rangequery(querybox)**  
  Returns all points within a given bounding box.

- **dump()**  
  Outputs the tree structure as JSON.

## Running the Program
Run using a trace file:

```bash
python test_ekd.py -tf <tracefile.csv>