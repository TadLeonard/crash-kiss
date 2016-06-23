#!/usr/bin/env python3

from kiss import parser


booth_group = parser.add_argument_group("booth options")
booth_group.add_argument("--photo-input-dir", default=".")
booth_group.add_argument("--crash-output-dir", default=".")
booth_group.add_argument("--last-crash-file", default="crash.mp4")


def main():
    args = parser.parse_args()


if __name__ == "__main__":
    main()
