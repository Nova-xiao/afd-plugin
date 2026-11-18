/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * License: CANN Open Software License Agreement Version 2.0
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: Define peer control geometry for per-reader input acknowledgements.
 * Note:
 * Create: 2026-11-18
 * History: 2026-11-18 introduce per-reader TAKEN slots.
 */
#pragma once
#include <stdint.h>

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#define MEGAMOE_LAYOUT_INLINE __forceinline__[aicore]
#else
#define MEGAMOE_LAYOUT_INLINE inline
#endif

namespace CamMegamoe {
constexpr int64_t SPLIT_CACHELINE_BYTES = 64;
constexpr int64_t SPLIT_TAKEN_SLOTS = 192;
constexpr int64_t SPLIT_PEER_BYTES = SPLIT_CACHELINE_BYTES * (1 + SPLIT_TAKEN_SLOTS);
constexpr int64_t SPLIT_MIN_PEER_REGION_BYTES = 48 * 1024;
constexpr int64_t SPLIT_PROGRESS_BYTES = 12 * 1024;
MEGAMOE_LAYOUT_INLINE constexpr int64_t SplitProgressOffset(int64_t ranks)
{
    return ranks * SPLIT_PEER_BYTES > SPLIT_MIN_PEER_REGION_BYTES
        ? ranks * SPLIT_PEER_BYTES : SPLIT_MIN_PEER_REGION_BYTES;
}
MEGAMOE_LAYOUT_INLINE constexpr int64_t SplitControlBytes(int64_t ranks)
{
    return (SplitProgressOffset(ranks) + SPLIT_PROGRESS_BYTES + 511) / 512 * 512;
}
MEGAMOE_LAYOUT_INLINE constexpr int64_t SplitTakenOffset(int64_t peer, int64_t slot)
{
    return peer * SPLIT_PEER_BYTES + (slot + 1) * SPLIT_CACHELINE_BYTES;
}
}

#undef MEGAMOE_LAYOUT_INLINE
