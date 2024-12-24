// =======================================
// You need to finish this module
// =======================================

module v_alu #(
    parameter SEW       = 32,
    parameter VLMAX     = 8,
    parameter VALUOP_DW = 5,
    parameter VREG_DW   = 256,
    parameter VREG_AW   = 5
)(
    input                   clk,
    input                   rst,
    input [VALUOP_DW-1:0]   valu_opcode_i,
    input [VREG_DW-1:0]     operand_v1_i,
    input [VREG_DW-1:0]     operand_v2_i,
    output[VREG_DW-1:0]     valu_result_o
);

localparam VALU_OP_NOP  = 5'd0 ;
localparam VALU_OP_VADD = 5'd1 ;
localparam VALU_OP_VMUL = 5'd2 ;

generate
    for (genvar i = 0; i < 8; i = i + 1) begin
        wire [SEW-1:0] op1 = operand_v1_i[(i+1)*32-1:i*32];
        wire [SEW-1:0] op2 = operand_v2_i[(i+1)*32-1:i*32];
        wire [SEW-1:0] out =    valu_opcode_i == VALU_OP_VADD ? (op1 + op2) : 
                                valu_opcode_i == VALU_OP_VMUL ? op1 * op2 : 'd0;
        assign valu_result_o[(i+1)*32-1:i*32] = out;
    end
endgenerate

endmodule
