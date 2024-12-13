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

reg [REG_DW-1:0] operand_1;
reg [REG_DW-1:0] operand_2;
reg [ALUOP_DW-1:0] alu_opcode;

assign operand_1_o = operand_1;
assign operand_2_o = operand_2;
assign alu_opcode_o = alu_opcode;

wire is_r_type = (inst_i[6:0] == `OPCODE_R_TYPE);
wire is_i_load = (inst_i[6:0] == `OPCODE_I_LOAD);
wire is_s_store = (inst_i[6:0] == `OPCODE_S_STORE);
wire is_b_type = (inst_i[6:0] == `OPCODE_B_TYPE);
wire is_u_lui = (inst_i[6:0] == `OPCODE_U_LUI);
wire is_j_jal = (inst_i[6:0] == `OPCODE_J_JAL);

assign rs1_en_o = is_r_type || is_i_load || is_s_store || is_b_type || is_u_lui || is_j_jal;
assign rs2_en_o = is_r_type || is_s_store || is_b_type || is_j_jal;
assign rs1_addr_o = inst_i[19:15];
assign rs2_addr_o = inst_i[24:20];
assign 

always @(posedge clk) begin
    if( rst == 1'b1) begin
        rs1_en_o <= 0;
        rs1_addr_o <= 0;
        rs2_en_o <= 0;
        rs2_addr_o <= 0;
        alu_opcode_o <= 0;
        operand_1_o <= 0;
        operand_2_o <= 0;
        branch_en_o <= 0;
        branch_offset_o <= 0;
        jump_en_o <= 0;
        jump_offset_o <= 0;
        mem_ren_o <= 0;
        mem_wen_o <= 0;
        mem_din_o <= 0;
        id_wb_en_o <= 0;
        id_wb_sel_o <= 0;
        id_wb_addr_o <= 0;
    end
end

endmodule 