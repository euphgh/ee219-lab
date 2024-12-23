; =======================================
; task1_2
; =======================================
lui     x1,     2148532224          ; x1 = 0x80100000(A base address)
lui     x2,     2150629376          ; x2 = 0x80300000(B base address)
lui     x3,     2152726528          ; x3 = 0x80500000(C base address)
lui     x4,     2154823680          ; x4 = 0x80700000(D base address)
addi    x8,     zero,   8           ; set const 8

addi    x5,     zero,   0           ; index i = 0
loop_i:
addi    x6,     zero,   0           ; index j = 0
loop_j:

add     x20,    x5,     zero        ; set i 
add     x21,    x6,     zero        ; set j 
add     x22,    x3,     zero        ; set base C
jal     x31,    get_addr            ; call
lw      x11,    0(x23)              ; x11  = C[i][j]

addi    x7,     zero,   0           ; index k = 0

mac:
add     x20,    x5,     zero        ; set i 
add     x21,    x7,     zero        ; set k 
add     x22,    x1,     zero        ; set base A
jal     x31,    get_addr            ; call
lw      x12,    0(x23)              ; x12  = B[i][k]

add     x20,    x7,     zero        ; set k
add     x21,    x6,     zero        ; set j 
add     x22,    x2,     zero        ; set base B
jal     x31,    get_addr            ; call
lw      x13,    0(x23)              ; x11  = B[k][j]

mul     x14,    x13,    x12         ; a * b
add     x11,    x11,    x14         ; c + a * b

addi    x7,     x7,     1           ; k = k + 1     
blt     x7,     x8,     mac         ; if (k < 8) goto mac

add     x20,    x5,     zero        ; set i 
add     x21,    x6,     zero        ; set j 
add     x22,    x4,     zero        ; set base D
jal     x31,    get_addr            ; call
sw      x11,    0(x23)              ; D[i][j] = result;

addi    x6,     x6,     1           ; j = j + 1     
blt     x6,     x8,     loop_j      ; if (j < 8) goto loop_j

addi    x5,     x5,     1           ; i = i + 1     
blt     x5,     x8,     loop_i      ; if (i < 8) goto loop_i

halt                                ;

; 20 = i, 21 = j, 22 = base, 23 = ret
get_addr:
mul     x23,    x20,    x8         ; i*8 
add     x23,    x23,    x21         ; i*8+j
addi    x24,    zero,   4
mul     x23,    x23,    x24         ; 4*(i*8+j)
add     x23,    x23,    x22         ; base+4*(i*8+j)
jalr    zero,   x31,    0           ; return