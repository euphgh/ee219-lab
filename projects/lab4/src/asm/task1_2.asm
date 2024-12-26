; =======================================
; task1_2
; =======================================

lui     x5,     2148532224              ; 0x80100000 Matrix A base addr
lui     x6,     2150629376              ; 0x80300000 Matrix B base addr
lui     x7,     2152726528              ; 0x80500000 Matrix C base addr 
lui     x8,     2154823680              ; 0x80700000 Matrix D base addr

addi    x1,     zero,   8               ; 8
addi    x4,     zero,   0               ; i = 0
addi    x28,    zero,   2               ; x28 = 2
row_loop:
    addi    x9,     zero,   0           ; j = 0
    col_loop:
        addi    x10,    zero,   0       ; k = 0
        addi    x11,    zero,   0       ; sum = 0
        
        k_loop:
            mul     x12,    x4,     x1      ; i*8
            add     x12,    x12,    x10     ; i*8 + k
            
            sll     x12,    x12,    x28     ; (i*8 + k)*4
            add     x12,    x5,     x12     ; addr of A[i][k]
            lw      x13,    0(x12)          ; load A[i][k]
            
            mul     x12,    x10,    x1      ; k*8
            add     x12,    x12,    x9      ; k*8 + j  
            sll     x12,    x12,    x28     ; (k*8 + j)*4
            add     x12,    x6,     x12     ; addr of B[k][j]
            lw      x14,    0(x12)          ; load B[k][j]
            
            mul     x15,    x13,    x14     ; A[i][k] * B[k][j]
            add     x11,    x11,    x15     ; sum += A[i][k] * B[k][j]
            
            addi    x10,    x10,    1       ; k++
            blt     x10,    x1,     k_loop  ; if k < 8 continue k_loop
        
        ; Add bias C[i][j]
        mul     x12,    x4,     x1          ; i*8
        add     x12,    x12,    x9          ; i*8 + j
        sll     x12,    x12,    x28         ; (i*8 + j)*4
        add     x12,    x7,     x12         ; addr of C[i][j]
        lw      x13,    0(x12)              ; load C[i][j]
        add     x11,    x11,    x13         ; sum += C[i][j]
        
        ; Store result to D[i][j]
        mul     x12,    x4,     x1          ; i*8
        add     x12,    x12,    x9          ; i*8 + j
        sll     x12,    x12,    x28         ; (i*8 + j)*4
        add     x12,    x8,     x12         ; addr of D[i][j]
        sw      x11,    0(x12)              ; D[i][j] = sum
        
        addi    x9,     x9,     1           ; j++
        blt     x9,     x1,     col_loop    ; if j < 8 continue col_loop
    
    addi    x4,     x4,     1               ; i++
    blt     x4,     x1,     row_loop        ; if i < 8 continue row_loop


halt                                    ;
