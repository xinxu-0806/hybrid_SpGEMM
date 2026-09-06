#ifndef ADAPTIVE_HBM_ROUTE_WORD_H
#define ADAPTIVE_HBM_ROUTE_WORD_H

// Shared Host/kernel layout for one route word.  Keep this header free of HLS
// types so the XRT Host and Vitis HLS compile from the same constants.
#define ADAPT_HBM_ROUTE_MODE_BITS 2U
#define ADAPT_HBM_ROUTE_MODE_MASK 3U
#define ADAPT_HBM_ROUTE_PRODUCTS_MAX 0x3fffffffU

#endif
