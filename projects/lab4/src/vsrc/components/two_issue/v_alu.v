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

reg [VREG_DW-1:0] result;
wire [SEW-1:0] v1_elements [VLMAX-1:0];
wire [SEW-1:0] v2_elements [VLMAX-1:0];
reg [SEW-1:0] result_elements [VLMAX-1:0];

genvar i;
generate
    for(i = 0; i < VLMAX; i = i + 1) begin
        assign v1_elements[i] = operand_v1_i[i*SEW +: SEW];
        assign v2_elements[i] = operand_v2_i[i*SEW +: SEW];
    end
endgenerate


integer j;
always @(*) begin
    if (rst) begin
        for(j = 0; j < VLMAX; j = j + 1) begin
            result_elements[j] = 0;
        end
    end else begin
        case(valu_opcode_i)
            VALU_OP_VADD: begin
                for(j = 0; j < VLMAX; j = j + 1) begin
                    result_elements[j] = v1_elements[j] + v2_elements[j];
                end
            end
            VALU_OP_VMUL: begin
                for(j = 0; j < VLMAX; j = j + 1) begin
                    result_elements[j] = v1_elements[j] * v2_elements[j];
                end
            end
            default: begin
                for(j = 0; j < VLMAX; j = j + 1) begin
                    result_elements[j] = 0;
                end
            end
        endcase
    end
end


generate
    for(i = 0; i < VLMAX; i = i + 1) begin
        always @(*) begin
            result[i*SEW +: SEW] = result_elements[i];
        end
    end
endgenerate

assign valu_result_o = result;

endmodule
