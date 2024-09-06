This work aims to replicate quantization from scratch, without using pytorch builtin functions. 
# Quantization Implementation To-Do List

This document outlines the tasks for implementing various quantization techniques for conventional models and large language models (LLMs). Each section below represents a different quantization approach and its associated techniques.

| Quantization                      | Description                                                                 | Status |
|-----------------------------------|-----------------------------------------------------------------------------|--------|
| **Weight Only: Static**           | Implement static quantization for weight-only models.                       | [X]    |
|                                   | Test and validate performance on sample models and performance metrics.     | [x]    |
| **Weight Only: Dynamic**          | Implement dynamic quantization for weight-only models.                      | [ ]    |
|                                   | Test and validate performance on sample models.                             | [ ]    |
| **Activation Only: Static**       | Implement static quantization for activation-only models.                   | [ ]    |
|                                   | Test and validate performance on sample models.                             | [ ]    |
| **Weight/Activation: Static**     | Implement static quantization for models with both weight and activation.   | [ ]    |
|                                   | Test and validate performance on sample models.                             | [ ]    |
| **Quantization Aware Training**   | Implement quantization-aware training for models.                           | [ ]    |
|                                   | Test and validate performance on sample models.                             | [ ]    |
| **AWQ**                           | Implement AWQ quantization method for models.                               | [ ]    |
|                                   | Test and validate performance on sample models.                             | [ ]    |
| **SmoothQuant**                   | Implement SmoothQuant quantization method for models.                       | [ ]    |
|                                   | Test and validate performance on sample models.                             | [ ]    |
| **OmniQuant**                     | Implement OmniQuant quantization method for models.                         | [ ]    |
|                                   | Test and validate performance on sample models.                             | [ ]    |
| **Quantized Kernels**             | Implement Quantized Kernels for layers.                                     | [ ]    |



## Notes
- Ensure compatibility with both conventional models and LLMs.
- Regularly update this list as new techniques or requirements emerge.
- Collaborate with team members to review and refine implementations.


This repo upon completion will be merged into sconce: https://github.com/satabios/sconce
