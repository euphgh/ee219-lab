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

wire [VREG_DW-1:0] operand_v1;
wire [VREG_DW-1:0] operand_v2;
wire [VALUOP_DW-1:0] valu_opcode;

assign operand_v1_o = rst ? 0 : operand_v1;
assign operand_v2_o = rst ? 0 : operand_v2;
assign valu_opcode_o = rst ? 0 : valu_opcode;

/*
In order to simplify the experiment, this experiment only needs to support the access mode of unit-stride . In addition, the nf , mew ,
mop , and lumop bits of the access instruction can be set to the 
default value of 0.
*/

wire [5:0] funct6 = inst_i[31:26];
wire [2:0] funct3 = inst_i[14:12];
wire [4:0] vs3 = inst_i[11:7];
wire [4:0] vs2 = inst_i[24:20];
wire [4:0] vs1 = inst_i[19:15];
wire [4:0] rs1 = inst_i[19:15];
wire [4:0] vd = inst_i[11:7];
wire [2:0] vwidth = inst_i[14:12];
wire [4:0] imm = inst_i[19:15];

wire is_vle32 = (inst_i[6:0] == `OPCODE_VL) && (funct6 == `FUNCT6_VLE32);
wire is_vse32 = (inst_i[6:0] == `OPCODE_VS) && (funct6 == `FUNCT6_VSE32);
wire is_vadd = (inst_i[6:0] == `OPCODE_VEC) && (funct6 == `FUNCT6_VADD);
wire is_vmul = (inst_i[6:0] == `OPCODE_VEC) && (funct6 == `FUNCT6_VMUL);
wire is_vv   = (inst_i[6:0] == `OPCODE_VEC) && (funct3 == 3'b000);
wire is_vx   = (inst_i[6:0] == `OPCODE_VEC) && (funct3 == 3'b100);
wire is_vi   = (inst_i[6:0] == `OPCODE_VEC) && (funct3 == 3'b011);

assign valu_opcode = 
    is_vadd ? VALU_OP_VADD :
    is_vmul ? VALU_OP_VMUL :
    VALU_OP_NOP;

// load, store 和 vx 类型的指令需要读 vs1 寄存器
assign rs1_en_o = !rst && (is_vle32 || is_vse32 || is_vx);
assign rs1_addr_o = rs1;



// IVV vi 和 vx 类型的指令不用读 vs1 寄存器
assign vs1_en_o = !rst && ((is_vadd || is_vmul) && (funct3 == 3'b000));
assign vs2_en_o = !rst && (is_vadd || is_vmul || is_vse32);
assign vs1_addr_o = vs1;
assign vs2_addr_o = is_vse32 ? vs3 : vs2;

assign vmem_ren_o = !rst && is_vle32;
assign vmem_wen_o = !rst && is_vse32;
assign vmem_addr_o = rs1_dout_i;
assign vmem_din_o = vs2_dout_i;         // 写入的值, 借用用 vs2 的端口

assign operand_v1 = is_vv ? vs1_dout_i : (is_vx ? {{(VREG_DW-REG_DW){rs1_dout_i[REG_DW-1]}}, rs1_dout_i} : (is_vi ? {{(VREG_DW-5){imm[4]}}, imm} : 0));
assign operand_v2 = vs2_dout_i;

assign vid_wb_en_o = !rst && (is_vadd || is_vmul || is_vle32);
assign vid_wb_sel_o = !rst && is_vle32;
assign vid_wb_addr_o = vd;

endmodule
