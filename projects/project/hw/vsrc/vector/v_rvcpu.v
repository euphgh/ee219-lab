`include "v_defines.v"

module v_rvcpu (
    input              clk,
    input              rst,
    input [`VINST_BUS] inst,

    input  [     `SREG_BUS] vec_rs1_data,
    output                  vec_rs1_r_ena,
    output [`SREG_ADDR_BUS] vec_rs1_r_addr,

    output                  vram_r_ena,
    output [`VRAM_ADDR_BUS] vram_r_addr,
    input  [`VRAM_DATA_BUS] vram_r_data,

    output                  vram_w_ena,
    output [`VRAM_ADDR_BUS] vram_w_addr,
    output [`VRAM_DATA_BUS] vram_w_data,
    output [`VRAM_DATA_BUS] vram_w_mask
);

  wire                  vwb_en;
  wire [`VREG_ADDR_BUS] vwb_addr;
  wire [     `VREG_BUS] vwb_data;
  wire                  vs1_en;
  wire [`VREG_ADDR_BUS] vs1_addr;
  wire                  vs2_en;
  wire [`VREG_ADDR_BUS] vs2_addr;

  // v_regfile Outputs
  wire [     `VREG_BUS] vs1_data;
  wire [     `VREG_BUS] vs2_data;

  v_regfile u_v_regfile (
      .clk       (clk),
      .rst       (rst),
      .vwb_en_i  (vwb_en),
      .vwb_addr_i(vwb_addr),
      .vwb_data_i(vwb_data),
      .vs1_en_i  (vs1_en),
      .vs1_addr_i(vs1_addr),
      .vs2_en_i  (vs2_en),
      .vs2_addr_i(vs2_addr),

      .vs1_data_o(vs1_data),
      .vs2_data_o(vs2_data)
  );

  wire [   `ALU_OP_BUS] valu_opcode;
  wire [     `VREG_BUS] operand_v1;
  wire [     `VREG_BUS] operand_v2;
  wire [`VMEM_ADDR_BUS] vmem_addr;
  wire                  vid_wb_en;
  wire                  vid_wb_sel;
  wire [`VREG_ADDR_BUS] vid_wb_addr;

  v_inst_decode u_v_inst_decode (
      .clk       (clk),
      .rst       (rst),
      .inst      (inst),
      .rs1_dout_i(vec_rs1_data),
      .vs1_dout_i(vs1_data),
      .vs2_dout_i(vs2_data),

      .rs1_en_o     (vec_rs1_r_ena),
      .rs1_addr_o   (vec_rs1_r_addr),
      .vs1_en_o     (vs1_en),
      .vs1_addr_o   (vs1_addr),
      .vs2_en_o     (vs2_en),
      .vs2_addr_o   (vs2_addr),
      .valu_opcode_o(valu_opcode),
      .operand_v1_o (operand_v1),
      .operand_v2_o (operand_v2),
      .vmem_ren_o   (vram_r_ena),
      .vmem_wen_o   (vram_w_ena),
      .vmem_addr_o  (vmem_addr),
      .vmem_din_o   (vram_w_data),
      .vid_wb_en_o  (vid_wb_en),
      .vid_wb_sel_o (vid_wb_sel),
      .vid_wb_addr_o(vid_wb_addr)
  );

  assign vram_r_addr = vmem_addr;
  assign vram_w_addr = vmem_addr;


  wire [`VREG_BUS] valu_result;

  v_execute u_v_alu (
      .clk          (clk),
      .rst          (rst),
      .valu_opcode_i(valu_opcode),
      .operand_v1_i (operand_v1),
      .operand_v2_i (operand_v2),

      .valu_result_o(valu_result)
  );

  v_write_back u_v_wb (
      .clk          (clk),
      .rst          (rst),
      .vid_wb_en_i  (vid_wb_en),
      .vid_wb_sel_i (vid_wb_sel),
      .vid_wb_addr_i(vid_wb_addr),
      .valu_result_i(valu_result),
      .vmem_result_i(vram_r_data),

      .vwb_en_o  (vwb_en),
      .vwb_addr_o(vwb_addr),
      .vwb_data_o(vwb_data)
  );

endmodule
