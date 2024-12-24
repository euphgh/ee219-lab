// =======================================
// You need to finish this module
// =======================================

`include "define_rv32v.v"

module v_id #(
    parameter VLMAX     = 8,
    parameter VALUOP_DW = 5,
    parameter VMEM_DW   = 256,
    parameter VMEM_AW   = 32,
    parameter VREG_DW   = 256,
    parameter VREG_AW   = 5,
    parameter INST_DW   = 32,
    parameter REG_DW    = 32,
    parameter REG_AW    = 5
) (
    input                   clk,
    input                   rst,

    input   [INST_DW-1:0]   inst_i,

    output                  rs1_en_o,
    output  [REG_AW-1:0]    rs1_addr_o,
    input   [REG_DW-1:0]    rs1_dout_i,

    output                  vs1_en_o,
    output  [VREG_AW-1:0]   vs1_addr_o,
    input   [VREG_DW-1:0]   vs1_dout_i,

    output                  vs2_en_o,
    output  [VREG_AW-1:0]   vs2_addr_o,
    input   [VREG_DW-1:0]   vs2_dout_i,

    output  [VALUOP_DW-1:0] valu_opcode_o,
    output  [VREG_DW-1:0]   operand_v1_o,
    output  [VREG_DW-1:0]   operand_v2_o,

    output                  vmem_ren_o,
    output                  vmem_wen_o,
    output  [VMEM_AW-1:0]   vmem_addr_o,
    output  [VMEM_DW-1:0]   vmem_din_o,

    output                  vid_wb_en_o,
    output                  vid_wb_sel_o,
    output  [VREG_AW-1:0]   vid_wb_addr_o
);

localparam VALU_OP_NOP  = 5'd0 ;
localparam VALU_OP_VADD = 5'd1 ;
localparam VALU_OP_VMUL = 5'd2 ;

wire [6:0] opcode   = inst_i[6:0];
wire [4:0] vd       = inst_i[11:7];
wire [4:0] vs3      = inst_i[11:7];
wire [2:0] width    = inst_i[14:12];
wire [4:0] rs1      = inst_i[19:15];
wire [4:0] vs1      = inst_i[19:15];
wire [4:0] vs2      = inst_i[24:20];
wire [2:0] funct3   = inst_i[14:12];
wire [5:0] funct6   = inst_i[31:26];

wire isLoad     = opcode == `OPCODE_VL;
wire isStore    = opcode == `OPCODE_VS;
wire isAdd      = opcode == `OPCODE_VEC && funct6 == `FUNCT6_VADD;
wire isMul      = opcode == `OPCODE_VEC && funct6 == `FUNCT6_VMUL;
wire isVV       = funct3 == `FUNCT3_IVV;
wire isVX       = funct3 == `FUNCT3_IVX;
wire isVI       = funct3 == `FUNCT3_IVI;

assign rs1_en_o = isLoad || isStore || isVX;
assign rs1_addr_o = rs1;

assign vs1_en_o = isVV;
assign vs1_addr_o = vs1;

assign vs2_en_o = !(isLoad);
assign vs2_addr_o = isStore ? vs3 : vs2;

assign valu_opcode_o =  (isAdd) ? VALU_OP_VADD :
                        (isMul) ? VALU_OP_VMUL : VALU_OP_NOP;
assign operand_v1_o =   isVV ? vs1_dout_i : 
                        isVI ? {8{{27{inst_i[19]}}, inst_i[19:15]}} :
                        isVX ? {8{rs1_dout_i}} : 0;
assign operand_v2_o =   vs2_dout_i;

assign vmem_ren_o = isLoad;
assign vmem_wen_o = isStore;
assign vmem_addr_o = rs1_dout_i;
assign vmem_din_o = vs2_dout_i;

assign vid_wb_en_o = (isLoad || isAdd || isMul);
assign vid_wb_sel_o = isLoad;
assign vid_wb_addr_o = vd;

endmodule

