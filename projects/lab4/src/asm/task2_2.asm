; =======================================
; task2_2
; =======================================

lui     x1,     2148532224          ; nop                                   ; x1 = 0x80100000(A base address)
lui     x2,     2151677952          ; nop                                   ; x2 = 0x80400000(Bt base address)
lui     x3,     2152726528          ; nop                                   ; x3 = 0x80500000(C base address)
lui     x4,     2154823680          ; nop                                   ; x4 = 0x80700000(D base address)
addi    x8,     zero,       8       ; nop                                   ; x8 = 8 const
addi    x9,     zero,       4       ; nop                                   ; x9 = 4 const
lui     x10,    2149580800          ; nop                                   ; x10 = 0x80200000(free space)

addi    x5,     zero,   0           ; nop                                   ; index i = 0
loop_i:

addi    x6,     zero,   0           ; nop                                   ; index j = 0
loop_j:

add     x24,    x5,     zero        ; nop                                   ; set i 
add     x25,    x1,     zero        ; nop                                   ; set base A
jal     x31,    line_addr           ; nop                                   ; call
nop                                 ; vle32.v   vx1,    x26,            1   ; vx1 = A[i, :]
add     x24,    x6,     zero        ; nop                                   ; set j 
add     x25,    x2,     zero        ; nop                                   ; set base Bt
jal     x31,    line_addr           ; nop                                   ; call
nop                                 ; vle32.v   vx2,    x26,            1   ; vx1 = Bt[j, :]

# calculate mul, store result to free space
nop                                 ; vmul.vv   vx3,    vx2,    vx1,    1   ; vx3 = vx1 * vx2;
nop                                 ; vse32.v   vx3,    x10,            1   ; free[0] = vx3;

add     x20,    x5,     zero        ; nop                                   ; set i 
add     x21,    x6,     zero        ; nop                                   ; set j 
add     x22,    x3,     zero        ; nop                                   ; set base C
jal     x31,    get_addr            ; nop                                   ; call
lw      x11,    0(x23)              ; nop                                   ; x11  = C[i][j]

addi    x7,     zero,   0           ; nop                                   ; index k = 0

acc:
mul     x12,    x7,     x9          ; nop                                   ; x12 = k * 4;
add     x12,    x12,    x10         ; nop                                   ; x12 = free + k * 4;
lw      x13,    0(x12)              ; nop                                   ; x13 = free[k * 4]
add     x11,    x11,    x13         ; nop                                   ; x11 += x13

addi    x7,     x7,     1           ; nop                                   ; k = k + 1     
blt     x7,     x8,     acc         ; nop                                   ; if (k < 8) goto acc

add     x20,    x5,     zero        ; nop                                   ; set i 
add     x21,    x6,     zero        ; nop                                   ; set j 
add     x22,    x4,     zero        ; nop                                   ; set base D
jal     x31,    get_addr            ; nop                                   ; call
sw      x11,    0(x23)              ; nop                                   ; D[i][j] = result;

addi    x6,     x6,     1           ; nop                                   ; j = j + 1
blt     x6,     x8,     loop_j      ; nop                                   ; if (j < 8) goto loop_j

addi    x5,     x5,     1           ; nop                                   ; i = i + 1     
blt     x5,     x8,     loop_i      ; nop                                   ; if (i < 8) goto loop_i

halt                                ; nop                                   ;


# 20 = i, 21 = j, 22 = base, 23 = ret
get_addr:
mul     x23,    x20,    x8          ; nop                                   ; # i*8 
add     x23,    x23,    x21         ; nop                                   ; # i*8+j
mul     x23,    x23,    x9          ; nop                                   ; 4*(i*8+j)
add     x23,    x23,    x22         ; nop                                   ; base+4*(i*8+j)
jalr    zero,   x31,    0           ; nop                                   ; return

# 24 = i, 25 = base, 26 = ret
line_addr:
addi    x26,    zero,   32          ; nop                                   ; x23 = 32(4*8)
mul     x26,    x24,    x26         ; nop                                   ; i*32 
add     x26,    x26,    x25         ; nop                                   ; base + i * 32
jalr    zero,   x31,    0           ; nop                                   ; return