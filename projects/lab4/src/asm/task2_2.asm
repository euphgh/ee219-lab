; =======================================
; task2_2
; =======================================

lui     x5,     2148532224              ; nop                                     ; 0x80100000 Matrix A base addr
lui     x6,     2151677952              ; nop                                     ; 0x80400000 Matrix B^T base addr
lui     x7,     2152726528              ; nop                                     ; 0x80500000 Matrix C base addr 
lui     x8,     2154823680              ; nop                                     ; 0x80700000 Matrix D base addr
lui     x13,    2154823680              ; nop                                     ; pointer point to D

; 原来的一行乘一列，现在可以用 vmul.vv 实现，将 a 和 b^T 中的每一行逐个相乘即可，第 i 行 x 第 j 行即为 c[i][j]
; 但要怎么实现累加呢？可以将向量运算的结果写回内存，然后读取内存中的值进行累加
; 其它部分按照 task1_2.asm 的写法即可，这样可以将矩阵乘复杂度从 O(n^3) 降到 O(n^2)
; 但是这个实验没提供向量累加的指令，所以还是得一个个加，复杂度没变啊
; 注意双发射的跳转指令需要跳转到 8 的倍数位置

addi    x1,     zero,   8               ; nop                                     ; row counter = 8
addi    x2,     zero,   8               ; nop                                     ; col counter = 8


addi    x4,     zero,   0               ; nop                                     ; i = 0
addi    x28,    zero,   8               ; nop                                     ; sum counter = 8

row_loop:
    addi    x9,     zero,   0           ; nop                                     ; j = 0
    lui     x6,     2151677952          ; nop                                     ; 0x80400000 Matrix B^T base addr

    col_loop:
        lui     x3,     2157969408              ;  nop                            ; 0x80a00000, 用于存放需要累加的数据
        nop                                     ;  vle32.v  vx1, x5, 1            ; load a[i][0:7]
        addi    x6,    x6,    32                ;  vle32.v  vx2, x6, 1            ; load b^T[j][0:7], x6 指向下一行
        addi    x9,    x9,    1                 ;  vmul.vv  vx3, vx1, vx2, 1      ; vx3 = a[i][0:7] * b^T[j][0:7], j++
        nop                                     ;  vse32.v  vx3, x3, 1            ; store vx3 to M[x3]
        addi    x10,   zero,  0                 ;  nop                            ; k = 0
        addi    x11,   zero,  0                 ;  nop                            ; sum = 0

        sum_loop:
            lw      x12,   0(x3)                ;  nop                            ; load M[x3][k]
            add     x11,   x12,   x11           ;  nop                            ; sum += M[x3][k]
            addi    x10,   x10,   1             ;  nop                            ; k++
            addi    x3,    x3,    4             ;  nop                            ; x3 += 4
            blt     x10,   x28,   sum_loop      ;  nop                            ; if k < 8 continue sum_loop

        sw      x11,    0(x13)                  ;  nop                            ; *D = sum
        addi    x13,    x13,    4               ;  nop                            ; *D++
        blt     x9,    x2,    col_loop          ;  nop                            ; if j < 8 continue row_loop

    addi    x4,    x4,    1             ; vle32.v   vx1, x8,  1      ;   load D[i][0:7], i++
    addi    x7,    x7,    32            ; vle32.v   vx2, x7,  1      ;   load C[i][0:7],  C 指向下一行
    addi    x5,    x5,    32            ; vadd.vv   vx3, vx1, vx2, 1 ;   D[i][0:7] += C[i][0:7],  A 指向下一行
    addi    x8,    x8,    32            ; vse32.v   vx3, x8,  1      ;   store D[i][0:7], D 指向下一行
    blt     x4,    x1,    row_loop      ;  nop                       ; if i < 8 continue row_loop


halt                                    ; nop ;
