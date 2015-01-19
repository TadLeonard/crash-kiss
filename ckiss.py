#!/usr/bin/env python

import argparse
from crash_kiss.edge import find_foreground


parser = argparse.ArgumentParser(
	description="Crash two faces into each other",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("targets", nargs="+")


def main():
        

	
