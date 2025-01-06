`include "v_defines.v"

module v_inst_decode (
    input              clk,
    input              rst,
    input [`VINST_BUS] inst,

    output                  rs1_en_o,
    output [`SREG_ADDR_BUS] rs1_addr_o,
    input  [     `SREG_BUS] rs1_dout_i,

    output                  vs1_en_o,
    output [`VREG_ADDR_BUS] vs1_addr_o,
    input  [     `VREG_BUS] vs1_dout_i,

    output                  vs2_en_o,
    output [`VREG_ADDR_BUS] vs2_addr_o,
    input  [     `VREG_BUS] vs2_dout_i,

    output [`ALU_OP_BUS] valu_opcode_o,
    output [  `VREG_BUS] operand_v1_o,
    output [  `VREG_BUS] operand_v2_o,

    output                  vmem_ren_o,
    output                  vmem_wen_o,
    output [`VMEM_ADDR_BUS] vmem_addr_o,
    output [`VMEM_DATA_BUS] vmem_din_o,

    output                  vid_wb_en_o,
    output                  vid_wb_sel_o,
    output [`VREG_ADDR_BUS] vid_wb_addr_o
);

endmodule
