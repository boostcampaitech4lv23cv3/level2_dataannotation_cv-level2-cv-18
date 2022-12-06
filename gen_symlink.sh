#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
LBLUE='\033[1;34m'
NC='\033[0m' # No Color

if [ "$#" -ne 1 ]; then
    echo ""
    echo -e "${LBLUE}EASY Symlink${NC} for ${RED}Light Observer${NC}"
    echo ""
    echo -e "Usage: ${GREEN}./gen_symlink.sh${NC} ${BLUE}SOURCE_FILE${NC}"
    echo ""
    echo -e "   ex) ${GREEN}./gen_symlink.sh${NC} ${BLUE}train${YELLOW}_ver2${BLUE}.py${NC}"
    echo -e "       Generated ${BLUE}./my_dir/train.py${NC} ← ${YELLOW}./my_dir/train_ver2.py${NC}"
    echo ""
fi

filename=$(basename -- "$1")
extension="${filename##*.}"
filename="${filename%.*}"
symlink_filename="${filename%_*}.$extension"

if [ -L "./my_dir/$symlink_filename" ]; then
    rm -f "./my_dir/$symlink_filename"
fi

if [ ! -f "./my_dir/$1" ]; then
    echo -e "${RED}File Not Found!${NC} [./my_dir/$1]"
else
    ln -s "$1" "./my_dir/$symlink_filename"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Success!${NC}"
    else
        echo -e "${RED}Exception!${NC}"
    fi
fi
