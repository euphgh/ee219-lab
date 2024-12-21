// =======================================
// 译码 define_rv32im.v 中定义的 OPCODE 所对应的指令
// =======================================


`include "define_rv32im.v"

module si_inst_decode #(
    parameter INST_DW   = 32,
    parameter INST_AW   = 32,
    parameter MEM_AW    = 32,
    parameter REG_DW    = 32,
    parameter REG_AW    = 5,
    parameter ALUOP_DW  = 5

) (
    input                   clk,
    input                   rst,
    // instruction
    input   [INST_DW-1:0]   inst_i,
    // regfile
    output                  rs1_en_o,
    output  [REG_AW-1:0]    rs1_addr_o,
    input   [REG_DW-1:0]    rs1_dout_i,
    output                  rs2_en_o,
    output  [REG_AW-1:0]    rs2_addr_o,
    input   [REG_DW-1:0]    rs2_dout_i,
    // alu
    output  [ALUOP_DW-1:0]  alu_opcode_o,
    output  [REG_DW-1:0]    operand_1_o,
    output  [REG_DW-1:0]    operand_2_o,
    output                  branch_en_o,
    output  [INST_AW-1:0]   branch_offset_o,
    output                  jump_en_o,
    output  [INST_AW-1:0]   jump_offset_o,
    // mem-access
    output                  mem_ren_o,
    output                  mem_wen_o,
    output  [INST_DW-1:0]   mem_din_o,
    // write-back
    output                  id_wb_en_o,
    output                  id_wb_sel_o,
    output  [REG_AW-1:0]    id_wb_addr_o 
);


localparam ALU_OP_NOP   = 5'd0 ;
localparam ALU_OP_ADD   = 5'd1 ;
localparam ALU_OP_MUL   = 5'd2 ;
localparam ALU_OP_BNE   = 5'd3 ;
localparam ALU_OP_JAL   = 5'd4 ;
localparam ALU_OP_LUI   = 5'd5 ;
localparam ALU_OP_AUIPC = 5'd6 ;
localparam ALU_OP_AND   = 5'd7 ;
localparam ALU_OP_SLL   = 5'd8 ;
localparam ALU_OP_SLT   = 5'd9 ;
localparam ALU_OP_BLT   = 5'd10 ;

wire [REG_DW-1:0] operand_1;
wire [REG_DW-1:0] operand_2;
wire [ALUOP_DW-1:0] alu_opcode;

assign operand_1_o = rst ? 0 : operand_1;
assign operand_2_o = rst ? 0 : operand_2;
assign alu_opcode_o = rst ? 0 : alu_opcode;

wire [2:0] funct3 = inst_i[14:12];
wire [6:0] funct7 = inst_i[31:25];
wire [31:0] imm_i = {{20{inst_i[31]}}, inst_i[31:20]};
wire [31:0] imm_s = {{20{inst_i[31]}}, inst_i[31:25], inst_i[11:7]};
wire [31:0] imm_b = {{20{inst_i[31]}}, inst_i[7], inst_i[30:25], inst_i[11:8], 1'b0};
wire [31:0] imm_u = {inst_i[31:12], 12'b0};
wire [31:0] imm_j = {11'b0, inst_i[31], inst_i[19:12], inst_i[20], inst_i[30:21], 1'b0};

wire is_r_type = (inst_i[6:0] == `OPCODE_R_TYPE);
wire is_i_common = (inst_i[6:0] == `OPCODE_I_COMMON);
wire is_i_load = (inst_i[6:0] == `OPCODE_I_LOAD);
wire is_s_store = (inst_i[6:0] == `OPCODE_S_STORE);
wire is_b_type = (inst_i[6:0] == `OPCODE_B_TYPE);
wire is_u_lui = (inst_i[6:0] == `OPCODE_U_LUI);
wire is_j_jal = (inst_i[6:0] == `OPCODE_J_JAL);

assign rs1_en_o = rst ? 0 : is_r_type || is_i_common || is_i_load || is_s_store || is_b_type;
assign rs2_en_o = rst ? 0 : is_r_type || is_s_store || is_b_type;
assign rs1_addr_o = rst ? 0 : inst_i[19:15];
assign rs2_addr_o = rst ? 0 : inst_i[24:20];

assign alu_opcode = 
    is_r_type ? (
        (funct3 == `FUNCT3_ADD && funct7 == `FUNCT7_ADD) ? ALU_OP_ADD :
        (funct3 == `FUNCT3_MUL && funct7 == `FUNCT7_MUL) ? ALU_OP_MUL :
        (funct3 == `FUNCT3_AND && funct7 == `FUNCT7_AND) ? ALU_OP_AND :
        (funct3 == `FUNCT3_SLL && funct7 == `FUNCT7_SLL) ? ALU_OP_SLL :
        ALU_OP_NOP
    ) : 
    is_i_common ? (
        (funct3 == `FUNCT3_SLTI) ? ALU_OP_SLT :
        (funct3 == `FUNCT3_ADDI) ? ALU_OP_ADD :
        ALU_OP_NOP
    ) : 
    is_i_load ? (
        (funct3 == `FUNCT3_LW) ? ALU_OP_ADD : ALU_OP_NOP
    ) : 
    is_s_store ? (
        (funct3 == `FUNCT3_SW) ? ALU_OP_ADD : ALU_OP_NOP
    ) : 
    is_b_type ? (
        (funct3 == `FUNCT3_BLT) ? ALU_OP_BLT : ALU_OP_NOP
    ) : 
    is_u_lui ? ALU_OP_LUI : 
    is_j_jal ? ALU_OP_JAL : 
    ALU_OP_NOP;

assign operand_1 = rst ? 0 : 
    is_r_type ? rs1_dout_i :
    is_i_common ? rs1_dout_i :
    is_i_load ? rs1_dout_i :
    is_s_store ? rs1_dout_i :
    is_b_type ? rs1_dout_i :
    is_u_lui ? imm_u :
    0;

assign operand_2 = rst ? 0 : 
    is_r_type ? rs2_dout_i :
    is_i_common ? imm_i :
    is_i_load ? imm_i :
    is_s_store ? imm_s :
    is_b_type ? rs2_dout_i :
    0;

assign branch_en_o = rst ? 0 : is_b_type;
assign branch_offset_o = rst ? 0 : 
    is_b_type ? imm_b :
    0;

assign jump_en_o = rst ? 0 : is_j_jal;
assign jump_offset_o = rst ? 0 : 
    is_j_jal ? imm_j :
    0;

assign mem_ren_o = rst ? 0 : is_i_load;
assign mem_wen_o = rst ? 0 : is_s_store;
assign mem_din_o = rst ? 0 : rs2_dout_i;

assign id_wb_en_o = rst ? 0 : is_r_type || is_i_common || is_i_load || is_u_lui || is_j_jal;
assign id_wb_sel_o = rst ? 0 : is_i_load;   // 选择 mem 或 alu result进行写回
assign id_wb_addr_o = rst ? 0 : inst_i[11:7]; //  rd 编号 

endmodule 
