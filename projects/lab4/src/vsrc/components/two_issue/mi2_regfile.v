// =======================================
// You need to finish this module
// =======================================

module mi2_regfile #(
    parameter REG_DW    = 32,
    parameter REG_AW    = 5
)(
    input                       clk,
    input                       rst,

    input                       is1_wb_en_i,
    input       [REG_AW-1:0]    is1_wb_addr_i,
    input       [REG_DW-1:0]    is1_wb_data_i,
    input                       is1_rs1_en_i,
    input       [REG_AW-1:0]    is1_rs1_addr_i,
    output reg  [REG_DW-1:0]    is1_rs1_data_o,
    input                       is1_rs2_en_i,
    input       [REG_AW-1:0]    is1_rs2_addr_i,
    output reg  [REG_DW-1:0]    is1_rs2_data_o,

    input                       is2_rs1_en_i,
    input       [REG_AW-1:0]    is2_rs1_addr_i,
    output reg  [REG_DW-1:0]    is2_rs1_data_o
);

integer i;
reg [REG_DW-1:0] regfile [2**REG_AW-1:0];

// vector 只读 rs1

always @(posedge clk) begin
    if (rst) begin
        for(i = 0; i < 2**REG_AW; i = i + 1) begin
            regfile[i] <= 0;
        end
    end else begin
        if (is1_wb_en_i && is1_wb_addr_i != 0) begin
            regfile[is1_wb_addr_i] <= is1_wb_data_i;
            $display("write is1_wb_data_i = %h", is1_wb_data_i);
        end
    end
end


always @(*) begin
    if (rst) begin
        is1_rs1_data_o = 0;
        is1_rs2_data_o = 0;
    end else begin
        if (is1_rs1_en_i) begin
            is1_rs1_data_o = regfile[is1_rs1_addr_i];
            $display("read is1_rs1_data_o = %h", is1_rs1_data_o);
        end else begin
            is1_rs1_data_o = 0;
        end
        
        if (is1_rs2_en_i) begin
            is1_rs2_data_o = regfile[is1_rs2_addr_i];
            $display("read is1_rs2_data_o = %h", is1_rs2_data_o);
        end else begin
            is1_rs2_data_o = 0;
        end
    end
end


always @(*) begin
    if (rst) begin
        is2_rs1_data_o = 0;
    end else begin
        if (is2_rs1_en_i) begin
            is2_rs1_data_o = regfile[is2_rs1_addr_i];
            $display("read is2_rs1_data_o = %h", is2_rs1_data_o);
        end else begin
            is2_rs1_data_o = 0;
        end
    end
end

endmodule
