// =======================================
// You need to finish this module
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

wire [6:0] opcode   = inst_i[6:0];
wire [4:0] rd       = inst_i[11:7];
wire [2:0] funct3   = inst_i[14:12];
wire [4:0] rs1      = inst_i[19:15];
wire [4:0] rs2      = inst_i[24:20];
wire [6:0] funct7   = inst_i[31:25];
wire isLoad     = opcode == 7'b0000011;
wire isStore    = opcode == 7'b0100011;
wire isBranch   = opcode == 7'b1100011;
wire isLUI      = opcode == 7'b0110111;
wire isJAL      = opcode == 7'b1101111;
wire isImm      = opcode == 7'b0010011;
wire isReg      = opcode == 7'b0110011;
wire isMUL      = isReg && funct3 == 3'b0 && funct7 == 7'b1;

assign rs1_en_o = !(isLUI || isJAL);
assign rs1_addr_o = rs1;

assign rs2_en_o = !(isImm || isLoad || isLUI || isJAL);
assign rs2_addr_o = rs2;

wire [4:0] aluType_op = (funct3 == `FUNCT3_ADD)  ? ALU_OP_ADD : 
                        (funct3 == `FUNCT3_AND)  ? ALU_OP_AND : 
                        (funct3 == `FUNCT3_SLL)  ? ALU_OP_SLL :
                        (funct3 == `FUNCT3_SLTI) ? ALU_OP_SLT : ALU_OP_NOP;

assign alu_opcode_o =   (isReg || isImm)  ? (isMUL ? ALU_OP_MUL : aluType_op) : 
                        (isLUI)  ? ALU_OP_LUI : 
                        (isJAL)  ? ALU_OP_JAL : 
                        (isBranch)  ? ALU_OP_BLT : 
                        (isLoad || isStore)  ? ALU_OP_ADD : ALU_OP_NOP;

assign operand_1_o = isLUI ? (inst_i & ~32'hfff) : rs1_dout_i;
assign operand_2_o = (isImm || isLoad) ? {{21{inst_i[31]}}, inst_i[30:20]} : 
                        isStore ? {{21{inst_i[31]}}, inst_i[30:25], inst_i[11:7]} : rs2_dout_i;

assign jump_en_o = isJAL;
assign jump_offset_o = {{12{inst_i[31]}}, inst_i[19:12], inst_i[20], inst_i[30:21], 1'b0};

assign branch_en_o = isBranch;
assign branch_offset_o = {{20{inst_i[31]}}, inst_i[7], inst_i[30:25], inst_i[11:8], 1'b0};

assign mem_ren_o = isLoad;
assign mem_wen_o = isStore;
assign mem_din_o = rs2_dout_i;


assign id_wb_en_o = !(isStore || isBranch);
assign id_wb_sel_o = isLoad;
assign id_wb_addr_o = rd;
                        
endmodule 
