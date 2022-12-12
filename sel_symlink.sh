#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
LBLUE='\033[1;34m'
NC='\033[0m' # No Color

if [ "$#" -ne 1 ] || ([ $1 != "last" ] && [ $1 != "best" ])
then
    echo ""
    echo -e "${LBLUE}EASY Symlink${NC} for ${RED}Light Observer${NC}"
    echo ""
    if [ ! -f "./trained_models/latest.pth" ]; then
        echo -e "${RED}File Not Found!  [latest.pth]${NC}"
    else
	dest_filename=`readlink -f "./trained_models/latest.pth"`
	dest_filename=`basename "${dest_filename}"`
	echo -e "Now selected [${YELLOW}${dest_filename}${NC}]"
    fi
    echo ""
    echo -e "Usage: ${GREEN}./sel_symlink.sh${NC} ${BLUE}[last | best]${NC}"
    echo ""
    echo -e "   ex) ${GREEN}./sel_symlink.sh${NC} ${BLUE}best${NC}"
    echo -e "       Changed ${BLUE}./trained_models/latest.pth${NC} ← ${YELLOW}./trained_models/best_model.pth${NC}"
    echo -e "       Submitting ${YELLOW}best_model.pth${NC}"
    echo ""
    exit
fi

if [ $1 == "last" ]; then
    if [ ! -f "./trained_models/swap.pth" ]; then
        echo -e "${RED}Isn't it already the latest? File Not Found! [swap.pth]${NC}"
        exit
    fi

    dest_filename=`readlink -f "./trained_models/swap.pth"`
    dest_filename=`basename "${dest_filename}"`
    rm -f "./trained_models/latest.pth"
    ln -s ${dest_filename} "./trained_models/latest.pth"

    echo -e "${GREEN}Success!${NC}"
    exit
fi

if [ $1 == "best" ]; then
    if [ ! -f "./trained_models/best_model.pth" ]; then
        echo -e "${RED}Is the best epoch file created? File Not Found! [best_model.pth]${NC}"
        exit
    fi

    dest_filename=`readlink -f "./trained_models/best_model.pth"`
    dest_filename=`basename "${dest_filename}"`
    rm -f "./trained_models/latest.pth"
    ln -s ${dest_filename} "./trained_models/latest.pth"

    echo -e "${GREEN}Success!${NC}"
    exit
fi
