`include "v_defines.v"

module v_execute (
    input                clk,
    input                rst,
    input  [`ALU_OP_BUS] valu_opcode_i,
    input  [  `VREG_BUS] operand_v1_i,
    input  [  `VREG_BUS] operand_v2_i,
    output [  `VREG_BUS] valu_result_o
);

  localparam VALU_OP_NOP = 'd0;
  localparam VALU_OP_VADD = 'd1;
  localparam VALU_OP_VMUL = 'd2;

  generate
    for (genvar i = 0; i < 8; i = i + 1) begin
      wire [`SEW-1:0] op1 = operand_v1_i[(i+1)*`SEW-1:i*`SEW];
      wire [`SEW-1:0] op2 = operand_v2_i[(i+1)*`SEW-1:i*`SEW];
      wire [`SEW-1:0] out =    valu_opcode_i == VALU_OP_VADD ? (op1 + op2) : 
                                valu_opcode_i == VALU_OP_VMUL ? op1 * op2 : 'd0;
      assign valu_result_o[(i+1)*`SEW-1:i*`SEW] = out;
    end
  endgenerate

endmodule
