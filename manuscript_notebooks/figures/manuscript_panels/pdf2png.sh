#!/bin/bash

# Loop over all PDF files in the directory
for file in *.pdf; do
    # Convert each PDF to PNG using pdftoppm
    pdftoppm -png "$file" "${file%.pdf}"
done