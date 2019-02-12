//
//  Header.h
//  SimpleCNN
//
//  Created by lyj on 2019/02/07.
//  Copyright © 2019 lyj. All rights reserved.
//

#ifndef Header_h
#define Header_h

#include <cstdint>

uint32_t byteswap_uint32(uint32_t a)
{
    return ((((a >> 24) & 0xff) << 0) |
            (((a >> 16) & 0xff) << 8) |
            (((a >> 8) & 0xff) << 16) |
            (((a >> 0) & 0xff) << 24));
}

#endif /* Header_h */
