/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* fatcat — streaming MoE LLM engine for AMD Strix Halo.
 * Entry point (P0 smoke build). */
#include <stdio.h>

#include "fatcat/version.h"

int main(void)
{
    printf("fatcat %s — streaming MoE engine for Strix Halo (P0 scaffold)\n",
           FATCAT_VERSION_STRING);
    return 0;
}
